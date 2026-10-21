import { Injectable } from '@angular/core';
import type {
  HandLandmarker,
  HolisticLandmarker,
  HolisticLandmarkerResult,
  NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import { Point, SequenceFrame } from './libras.service';

// WASM servido pelo próprio app (copiado de node_modules em angular.json),
// na mesma versão do pacote. Modelos com versão fixa (/1/, não /latest/)
// para o app usar exatamente os arquivos da validação numérica (2.2.2).
const WASM_PATH = '/mediapipe/wasm';
const HOLISTIC_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task';
const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

/** 4 casas decimais bastam (1/10000 da imagem) e encolhem o payload. */
function round4(value: number): number {
  return Math.round(value * 10000) / 10000;
}

function xyz(landmarks: NormalizedLandmark[] | undefined): Point[] | null {
  return landmarks?.length ? landmarks.map(l => [round4(l.x), round4(l.y), round4(l.z)]) : null;
}

/**
 * MediaPipe Tasks rodando no navegador. O servidor recebe só landmarks
 * (docs/API_CONTRACT.md) — nenhuma imagem sai do navegador.
 *
 * Os frames vêm do <video> sem espelhar (o espelhamento do preview é só
 * CSS), igual aos vídeos de treino.
 */
@Injectable({ providedIn: 'root' })
export class MediapipeService {
  private holistic?: Promise<HolisticLandmarker>;
  private hands?: Promise<HandLandmarker>;

  /** Carrega o pacote sob demanda: fica fora do bundle inicial. */
  private async fileset() {
    const vision = await import('@mediapipe/tasks-vision');
    return { vision, files: await vision.FilesetResolver.forVisionTasks(WASM_PATH) };
  }

  getHolistic(): Promise<HolisticLandmarker> {
    this.holistic ??= this.fileset().then(({ vision, files }) =>
      vision.HolisticLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: HOLISTIC_MODEL_URL, delegate: 'GPU' },
        runningMode: 'VIDEO',
      })
    );
    return this.holistic;
  }

  getHands(): Promise<HandLandmarker> {
    this.hands ??= this.fileset().then(({ vision, files }) =>
      vision.HandLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 1,
      })
    );
    return this.hands;
  }

  /** Converte o resultado do HolisticLandmarker para o frame do contrato. */
  toSequenceFrame(result: HolisticLandmarkerResult): SequenceFrame {
    const pose = result.poseLandmarks[0];
    return {
      pose: pose?.length
        ? pose.map(l => [round4(l.x), round4(l.y), round4(l.z), round4(l.visibility ?? 0)])
        : null,
      face: xyz(result.faceLandmarks[0]),
      left_hand: xyz(result.leftHandLandmarks[0]),
      right_hand: xyz(result.rightHandLandmarks[0]),
    };
  }

  /** Landmarks [x, y] da primeira mão detectada, ou null. */
  async detectHand(video: HTMLVideoElement, timestampMs: number): Promise<Point[] | null> {
    const hands = await this.getHands();
    const result = hands.detectForVideo(video, timestampMs);
    const hand = result.landmarks[0];
    return hand?.length ? hand.map(l => [round4(l.x), round4(l.y)]) : null;
  }

  /** Landmarks de um frame de vídeo para /predict_sequence. */
  async detectFrame(video: HTMLVideoElement, timestampMs: number): Promise<SequenceFrame> {
    const holistic = await this.getHolistic();
    return this.toSequenceFrame(holistic.detectForVideo(video, timestampMs));
  }
}

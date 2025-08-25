import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { HandLandmarker, FilesetResolver, NormalizedLandmark } from '@mediapipe/tasks-vision';
import { Observable } from 'rxjs';

// --- FUNÇÃO DE NORMALIZAÇÃO AGORA É PÚBLICA ---
// Movemos a função para fora da classe para que outros componentes possam usá-la.
export function normalizeLandmarks(landmarks: NormalizedLandmark[]): { x: number; y: number }[] | null {
  if (landmarks.length < 21) return null;
  const wrist = landmarks[0];
  const middleFingerBase = landmarks[9];
  const scale = Math.sqrt(Math.pow(wrist.x - middleFingerBase.x, 2) + Math.pow(wrist.y - middleFingerBase.y, 2));
  if (scale === 0) return null;
  return landmarks.map(lm => ({ x: (lm.x - wrist.x) / scale, y: (lm.y - wrist.y) / scale }));
}


@Injectable({ providedIn: 'root' })
export class SignRecognitionService {
  public handLandmarker!: HandLandmarker; // Tornando público para a ferramenta de coleta
  private isInitialized = false;
  private apiUrl = 'http://localhost:3000/api';

  constructor(private http: HttpClient) { }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");
    this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: `assets/models/hand_landmarker.task`, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 1
    });
    this.isInitialized = true;
  }
  
  getLocalLandmarks(videoElement: HTMLVideoElement, timestamp: number): NormalizedLandmark[] | null {
    if (!this.isInitialized) return null;
    const results = this.handLandmarker.detectForVideo(videoElement, timestamp);
    return results.landmarks?.[0] || null;
  }
  
  recognizeSign(landmarks: NormalizedLandmark[]): Observable<{ letter: string }> {
    return this.http.post<{ letter: string }>(`${this.apiUrl}/recognize`, { landmarks });
  }
}
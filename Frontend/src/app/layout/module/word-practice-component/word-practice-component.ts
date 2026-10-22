import { Component, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { LibrasService, Prediction, SequenceFrame } from '../../../services/libras.service';
import { MediapipeService } from '../../../services/mediapipe.service';

// Mesmo ritmo do treino: os clipes têm ~4 s e são reamostrados para 30
// frames (≈ 7,5 fps). Amostrar mais rápido encurtaria a janela no tempo e o
// sinal chegaria "acelerado" ao modelo.
const SAMPLE_INTERVAL_MS = 1000 / 7.5;
// Janela enviada ao servidor = num_frames do modelo (30 frames ≈ 4 s).
const WINDOW_FRAMES = 30;
// Com a janela cheia, nova predição a cada STRIDE_FRAMES frames novos
// (janela deslizante) — uma requisição por ~1 s, não uma por frame.
const STRIDE_FRAMES = 8;

@Component({
  selector: 'app-word-practice',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatButtonModule,
    MatProgressBarModule,
    MatIconModule,
    MatProgressSpinnerModule
  ],
  templateUrl: './word-practice-component.html',
  styleUrls: ['./word-practice-component.scss']
})
export class WordPracticeComponent implements OnInit, OnDestroy {
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;

  videoStream!: MediaStream;
  videoWidth = 640;
  videoHeight = 480;

  availableWords: string[] = [];
  targetWord: string = '';
  predictions: Prediction[] | null = null;
  bufferProgress = 0;
  isReady = false;
  isCorrect = false;
  isLoading = true;
  feedbackMessage = 'Carregando o reconhecimento de sinais...';
  history: string[] = [];
  score = 0;

  // Buffer de frames fica no navegador; o servidor é stateless.
  private frames: SequenceFrame[] = [];
  private framesSinceRequest = 0;
  private requestInFlight = false;
  private detecting = false;
  private lastSampleAt = -Infinity;
  private animationFrameId?: number;
  private destroyed = false;

  constructor(
    private librasService: LibrasService,
    private mediapipe: MediapipeService,
  ) {}

  ngOnInit(): void {
    this.loadClasses();
    this.startWebcam();
  }

  ngOnDestroy(): void {
    this.destroyed = true;
    if (this.animationFrameId !== undefined) {
      cancelAnimationFrame(this.animationFrameId);
    }
    this.stopWebcam();
  }

  loadClasses(): void {
    this.librasService.getClasses().subscribe({
      next: (response) => {
        this.availableWords = response.classes;
        this.selectRandomWord();
      },
      error: (_err: any) => {
        this.availableWords = ['Sim', 'Oi', 'Obrigado', 'Por favor', 'Bom dia',
                               'Casa', 'Desculpa', 'Trabalho', 'De nada',
                               'Amigo', 'Família'];
        this.selectRandomWord();
      }
    });
  }

  startWebcam(): void {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(async stream => {
        this.videoStream = stream;
        const video = this.videoElement.nativeElement;
        video.srcObject = stream;
        await video.play();
        await this.mediapipe.getHolistic();
        this.isLoading = false;
        this.feedbackMessage = `Faça o sinal de "${this.targetWord}"`;
        this.processLoop();
      })
      .catch(err => {
        console.error('Erro ao iniciar a câmera ou o MediaPipe:', err);
        this.isLoading = false;
        this.feedbackMessage = 'Erro ao acessar a webcam ou carregar o reconhecimento.';
      });
  }

  stopWebcam(): void {
    if (this.videoStream) {
      this.videoStream.getTracks().forEach(track => track.stop());
    }
  }

  selectRandomWord(): void {
    if (this.availableWords.length === 0) return;
    const randomIndex = Math.floor(Math.random() * this.availableWords.length);
    this.targetWord = this.availableWords[randomIndex];
    this.predictions = null;
    this.isCorrect = false;
    this.resetBuffer();
    if (!this.isLoading) {
      this.feedbackMessage = `Faça o sinal de "${this.targetWord}"`;
    }
  }

  private resetBuffer(): void {
    this.frames = [];
    this.framesSinceRequest = 0;
    this.isReady = false;
    this.bufferProgress = 0;
  }

  /** Um detectForVideo a cada SAMPLE_INTERVAL_MS (ritmo do treino). */
  private processLoop = (): void => {
    if (this.destroyed) return;
    this.animationFrameId = requestAnimationFrame(this.processLoop);

    const video = this.videoElement?.nativeElement;
    const now = performance.now();
    if (!video || video.readyState < 2 || now - this.lastSampleAt < SAMPLE_INTERVAL_MS) return;
    if (this.detecting || this.isCorrect) return;

    this.lastSampleAt = now;
    this.detecting = true;
    this.mediapipe.detectFrame(video, now)
      .then(frame => this.pushFrame(frame))
      .catch(err => console.error('Erro no MediaPipe:', err))
      .finally(() => this.detecting = false);
  };

  private pushFrame(frame: SequenceFrame): void {
    this.frames.push(frame);
    if (this.frames.length > WINDOW_FRAMES) this.frames.shift();
    this.framesSinceRequest++;

    this.bufferProgress = (this.frames.length / WINDOW_FRAMES) * 100;
    this.isReady = this.frames.length === WINDOW_FRAMES;

    if (this.isReady && this.framesSinceRequest >= STRIDE_FRAMES && !this.requestInFlight) {
      // Sem mão em nenhum frame não há sinal para classificar (servidor daria 422).
      if (this.frames.some(f => f.left_hand || f.right_hand)) {
        this.sendWindow([...this.frames]);
      }
    }
  }

  private sendWindow(frames: SequenceFrame[]): void {
    this.requestInFlight = true;
    this.framesSinceRequest = 0;
    const word = this.targetWord;

    this.librasService.predictSequence(frames).subscribe({
      next: (response) => {
        this.requestInFlight = false;
        if (word !== this.targetWord || this.isCorrect) return;  // palavra trocou no meio

        this.predictions = response.predictions;
        const top1 = response.predictions[0];
        if (top1 && top1.word === this.targetWord && top1.confidence > 0.3) {
          this.isCorrect = true;
          this.score++;
          this.history.push(this.targetWord);
          this.feedbackMessage = `🎉 Correto! Você fez "${this.targetWord}"!`;
          setTimeout(() => this.selectRandomWord(), 2500);
        } else if (response.predictions.some(p => p.word === this.targetWord)) {
          this.feedbackMessage = `Quase! "${this.targetWord}" está entre as opções.`;
        }
      },
      error: (_err: any) => {
        this.requestInFlight = false;
        this.feedbackMessage = 'Erro na conexão com o servidor.';
      }
    });
  }

  clearHistory(): void {
    this.history = [];
    this.score = 0;
    this.selectRandomWord();
  }

  getConfidenceColor(confidence: number): string {
    if (confidence >= 0.7) return '#4caf50';
    if (confidence >= 0.4) return '#ff9800';
    return '#f44336';
  }

  getConfidencePercent(confidence: number): string {
    return (confidence * 100).toFixed(1);
  }
}

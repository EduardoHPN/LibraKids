import { Component, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { LetterPredictionResponse, LibrasService } from '../../../services/libras.service';
import { MediapipeService } from '../../../services/mediapipe.service';

// Uma classificação a cada 500 ms (como antes), agora enviando só os
// 21 landmarks da mão em vez da imagem.
const PREDICT_INTERVAL_MS = 500;

@Component({
  selector: 'app-libras-practice',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatButtonModule,
    MatProgressBarModule,
    MatIconModule,
    MatProgressSpinnerModule
  ],
  templateUrl: './alphabet-practice.component.html',
  styleUrls: ['./alphabet-practice.component.scss']
})
export class AlphabetPracticeComponent implements OnInit, OnDestroy {
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;

  videoStream!: MediaStream;
  videoWidth = 640;
  videoHeight = 480;
  targetLetters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'];
  targetLetter: string = '';
  predictionResult: Partial<LetterPredictionResponse> | null = null;
  processingInterval: any;
  isLoading = false;

  constructor(
    private librasService: LibrasService,
    private mediapipe: MediapipeService,
  ) { }

  ngOnInit(): void {
    this.startWebcam();
    this.selectRandomLetter();
  }

  ngOnDestroy(): void {
    this.stopWebcam();
    if (this.processingInterval) {
      clearInterval(this.processingInterval);
    }
  }

  startWebcam(): void {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(async stream => {
        this.videoStream = stream;
        const video = this.videoElement.nativeElement;
        video.srcObject = stream;
        await video.play();
        await this.mediapipe.getHands();
        this.startProcessing();
      })
      .catch(err => {
        console.error('Erro ao iniciar a câmera ou o MediaPipe:', err);
        alert('Erro ao acessar a webcam ou carregar o reconhecimento. Verifique as permissões.');
      });
  }

  stopWebcam(): void {
    if (this.videoStream) {
      this.videoStream.getTracks().forEach(track => track.stop());
    }
  }

  selectRandomLetter(): void {
    const randomIndex = Math.floor(Math.random() * this.targetLetters.length);
    this.targetLetter = this.targetLetters[randomIndex];
    this.predictionResult = null;
  }

  startProcessing(): void {
    if (this.processingInterval) clearInterval(this.processingInterval);
    this.processingInterval = setInterval(() => this.predictLetter(), PREDICT_INTERVAL_MS);
  }

  async predictLetter(): Promise<void> {
    const video = this.videoElement?.nativeElement;
    if (!video?.srcObject || video.readyState < 2 || this.isLoading) return;

    this.isLoading = true;
    let hand;
    try {
      hand = await this.mediapipe.detectHand(video, performance.now());
    } catch (err) {
      console.error('Erro no MediaPipe:', err);
      this.isLoading = false;
      return;
    }

    if (!hand) {
      // Sem mão no quadro: responde localmente, sem chamar o servidor.
      this.predictionResult = { message: 'Nenhuma mão detectada.' };
      this.isLoading = false;
      return;
    }

    const letter = this.targetLetter;
    this.librasService.predictLetterLandmarks(hand, letter).subscribe({
      next: (response) => {
        this.isLoading = false;
        if (letter !== this.targetLetter) return;  // letra trocou no meio
        this.predictionResult = response;
        if (response.correct) {
          setTimeout(() => this.selectRandomLetter(), 2000);
        }
      },
      error: (error: any) => {
        this.predictionResult = { message: 'Erro na conexão com o servidor.', success: false };
        this.isLoading = false;
        console.error(error);
      }
    });
  }
}

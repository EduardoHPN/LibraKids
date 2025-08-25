import { Component, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { SignRecognitionService } from '../../../services/sign-recognition';

@Component({
  selector: 'app-alphabet-practice',
  standalone: true,
  imports: [ CommonModule, RouterLink, MatCardModule, MatButtonModule, MatIconModule, MatProgressSpinnerModule ],
  templateUrl: './alphabet-practice.component.html',
  styleUrls: ['./alphabet-practice.component.scss']
})
export class AlphabetPracticeComponent implements OnInit, OnDestroy {
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;
  statusMessage: string = 'Carregando modelo de IA...';
  isCameraLoading: boolean = true;
  predictedLetter: string = '?';
  private stream: MediaStream | null = null;
  private isComponentActive = true;

  constructor(private signRecognitionService: SignRecognitionService) { }

  async ngOnInit(): Promise<void> {
    await this.signRecognitionService.initialize();
    this.statusMessage = 'Modelo carregado. Iniciando câmera...';
    await this.setupCamera();
  }

  ngOnDestroy(): void {
    this.isComponentActive = false;
    this.stopCamera();
  }

  async setupCamera(): Promise<void> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      this.videoElement.nativeElement.srcObject = this.stream;
      this.videoElement.nativeElement.addEventListener('loadeddata', () => {
        this.isCameraLoading = false;
        this.statusMessage = 'Posicione sua mão na área do vídeo.';
        this.predictionLoop();
      });
    } catch (err) {
      this.isCameraLoading = false;
      this.statusMessage = 'Erro ao acessar a câmera. Verifique as permissões.';
    }
  }

  predictionLoop(): void {
    if (!this.isComponentActive || this.videoElement.nativeElement.paused || this.videoElement.nativeElement.ended) {
      return;
    }
    const landmarks = this.signRecognitionService.getLocalLandmarks(this.videoElement.nativeElement, performance.now());
    if (landmarks) {
      this.signRecognitionService.recognizeSign(landmarks).subscribe(response => {
        this.predictedLetter = response.letter;
      });
    } else {
      this.predictedLetter = '?';
    }
    requestAnimationFrame(() => this.predictionLoop());
  }

  stopCamera(): void {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
    }
  }
}
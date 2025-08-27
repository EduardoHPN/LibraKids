import { Component, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { LibrasService } from '../../services/libras.service';

@Component({
  selector: 'app-libras-training',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatButtonModule,
    MatFormFieldModule,
    MatInputModule,
    MatIconModule,
    MatProgressSpinnerModule
  ],
  templateUrl: './data-collector.html',
  styleUrls: ['./data-collector.scss']
})
export class DataCollectorComponent implements OnInit, OnDestroy {
  @ViewChild('videoElement') videoElement!: ElementRef;
  @ViewChild('canvasElement') canvasElement!: ElementRef;

  videoStream!: MediaStream;
  videoWidth = 640;
  videoHeight = 480;
  trainingLabel = '';
  trainingImageCount = 0;
  isLoading = false;
  feedbackMessage = 'Posicione sua mão na frente da câmera para treinar o modelo.';

  constructor(private librasService: LibrasService) { }

  ngOnInit(): void {
    this.startWebcam();
  }

  ngOnDestroy(): void {
    this.stopWebcam();
  }

  startWebcam(): void {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        this.videoStream = stream;
        this.videoElement.nativeElement.srcObject = stream;
        this.videoElement.nativeElement.play();
      })
      .catch(err => {
        console.error('Erro ao acessar a webcam:', err);
        alert('Erro ao acessar a webcam. Verifique as permissões.');
      });
  }

  stopWebcam(): void {
    if (this.videoStream) {
      this.videoStream.getTracks().forEach(track => track.stop());
    }
  }

  captureAndSave(): void {
    if (!this.trainingLabel) {
      this.feedbackMessage = 'Por favor, insira a letra que você está treinando.';
      return;
    }

    const video = this.videoElement.nativeElement;
    const canvas = this.canvasElement.nativeElement;
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageDataUrl = canvas.toDataURL('image/jpeg');

    this.isLoading = true;
    this.librasService.recordData(imageDataUrl, this.trainingLabel).subscribe({
      next: (response) => {
        this.trainingImageCount++;
        this.feedbackMessage = `Imagem salva com sucesso! Total: ${this.trainingImageCount}`;
        this.isLoading = false;
      },
      error: (error) => {
        this.feedbackMessage = 'Erro ao salvar a imagem. Tente novamente.';
        this.isLoading = false;
      }
    });
  }
}
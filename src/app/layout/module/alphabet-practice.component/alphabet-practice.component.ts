// Remova tudo relacionado a 'isTrainingMode' e 'trainingLabel'

import { Component, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressBarModule } from '@angular/material/progress-bar'; // Corrigido
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner'; // Para o spinner
import { LibrasService } from '../../../services/libras.service';

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
  @ViewChild('videoElement') videoElement!: ElementRef;
  @ViewChild('canvasElement') canvasElement!: ElementRef;

  videoStream!: MediaStream;
  videoWidth = 640;
  videoHeight = 480;
  targetLetters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'];
  targetLetter: string = '';
  predictionResult: any = null;
  processingInterval: any;
  isLoading = false;

  constructor(private librasService: LibrasService) { }

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
      .then(stream => {
        this.videoStream = stream;
        this.videoElement.nativeElement.srcObject = stream;
        this.videoElement.nativeElement.play();
        setTimeout(() => this.startProcessing(), 1000);
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

  selectRandomLetter(): void {
    const randomIndex = Math.floor(Math.random() * this.targetLetters.length);
    this.targetLetter = this.targetLetters[randomIndex];
    this.predictionResult = null;
  }

  startProcessing(): void {
    if (this.processingInterval) clearInterval(this.processingInterval);
    this.processingInterval = setInterval(() => this.predictLetter(), 500);
  }

  predictLetter(): void {
    if (!this.videoElement.nativeElement.srcObject) return;
    const video = this.videoElement.nativeElement;
    const canvas = this.canvasElement.nativeElement;
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageDataUrl = canvas.toDataURL('image/jpeg');
    this.isLoading = true;
    this.librasService.predictLetter(imageDataUrl, this.targetLetter).subscribe({
      next: (response) => {
        this.predictionResult = response;
        this.isLoading = false;
        if (response.correct) {
          setTimeout(() => this.selectRandomLetter(), 2000);
        }
      },
      error: (error) => {
        this.predictionResult = { message: 'Erro na conexão com o servidor.', success: false };
        this.isLoading = false;
        console.error(error);
      }
    });
  }
}
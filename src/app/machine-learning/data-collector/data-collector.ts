import { Component, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { SignRecognitionService, normalizeLandmarks } from '../../services/sign-recognition';

interface CapturedData { [key: string]: { x: number, y: number }[][]; }

@Component({
  selector: 'app-data-collector',
  standalone: true,
  imports: [CommonModule, FormsModule, MatSelectModule, MatFormFieldModule, MatButtonModule, MatIconModule],
  templateUrl: './data-collector.html',
  styleUrls: ['./data-collector.scss']
})
export class DataCollectorComponent implements OnInit, OnDestroy {
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;
  
  statusMessage: string = 'Iniciando...';
  alphabet: string[] = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
  selectedLetter: string = 'A';
  capturedData: CapturedData = {};

  private stream: MediaStream | null = null;
  
  constructor(private signRecognitionService: SignRecognitionService) {}

  async ngOnInit(): Promise<void> {
    await this.signRecognitionService.initialize();
    await this.setupCamera();
  }

  ngOnDestroy(): void {
    this.stopCamera();
  }

  async setupCamera(): Promise<void> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      this.videoElement.nativeElement.srcObject = this.stream;
      this.statusMessage = 'Câmera pronta. Selecione uma letra e capture os exemplos.';
    } catch (err) {
      this.statusMessage = 'Erro ao acessar a câmera. Verifique as permissões.';
    }
  }

  captureSample(): void {
    const landmarks = this.signRecognitionService.getLocalLandmarks(this.videoElement.nativeElement, performance.now());
    if (landmarks) {
      const normalized = normalizeLandmarks(landmarks);
      if (normalized) {
        if (!this.capturedData[this.selectedLetter]) {
          this.capturedData[this.selectedLetter] = [];
        }
        this.capturedData[this.selectedLetter].push(normalized);
      }
    }
  }

  downloadData(): void {
    const dataStr = JSON.stringify(this.capturedData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', 'collected_data.json');
    linkElement.click();
  }

  getSampleCount(letter: string): number {
    return this.capturedData[letter]?.length || 0;
  }

  isDataCaptured(): boolean {
    return Object.keys(this.capturedData).length > 0;
  }

  stopCamera(): void {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
    }
  }
}
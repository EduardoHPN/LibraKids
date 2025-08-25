import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';

interface CapturedData { [key: string]: { x: number; y: number }[][] }
interface SignModel { [key: string]: { x: number; y: number }[] }

@Component({
  selector: 'app-model-processor',
  standalone: true,
  imports: [CommonModule, MatCardModule, MatButtonModule, MatIconModule, MatSnackBarModule],
  templateUrl: './model-processor.component.html',
  styleUrls: ['./model-processor.component.scss']
})
export class ModelProcessorComponent {
  modelCode: string = '';
  fileName: string = '';
  isDragOver = false;

  constructor(private snackBar: MatSnackBar) {}

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.isDragOver = true;
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    this.isDragOver = false;
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.isDragOver = false;
    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      this.processFile(files[0]);
    }
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.processFile(input.files[0]);
    }
  }

  private processFile(file: File): void {
    if (file.name !== 'collected_data.json' || file.type !== 'application/json') {
      this.snackBar.open('Erro: Por favor, selecione o ficheiro "collected_data.json".', 'Fechar', { duration: 3000 });
      return;
    }

    this.fileName = file.name;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data: CapturedData = JSON.parse(e.target?.result as string);
        const model = this.processData(data);
        const sortedModel = Object.keys(model).sort().map(key => ({ letter: key, landmarks: model[key] }));
        this.modelCode = `export const ALPHABET_MODEL_DATA = ${JSON.stringify(sortedModel, null, 2)};`;
      } catch (error) {
        this.snackBar.open('Erro ao processar o ficheiro JSON.', 'Fechar', { duration: 3000 });
      }
    };
    reader.readAsText(file);
  }

  private processData(data: CapturedData): SignModel {
    const finalModel: SignModel = {};
    for (const letter in data) {
        const samples = data[letter];
        const avgLandmarks = [];
        for (let i = 0; i < 21; i++) {
            let sumX = 0, sumY = 0, validSamples = 0;
            for (const sample of samples) {
                if(sample && sample[i]) {
                   sumX += sample[i].x;
                   sumY += sample[i].y;
                   validSamples++;
                }
            }
            if (validSamples > 0) {
                avgLandmarks.push({
                    x: parseFloat((sumX / validSamples).toFixed(5)),
                    y: parseFloat((sumY / validSamples).toFixed(5))
                });
            }
        }
        if (avgLandmarks.length === 21) {
           finalModel[letter] = avgLandmarks;
        }
    }
    return finalModel;
  }

  copyToClipboard(): void {
    if (this.modelCode) {
      navigator.clipboard.writeText(this.modelCode).then(() => {
        this.snackBar.open('Código do modelo copiado para a área de transferência!', 'Fechar', { duration: 2000 });
      });
    }
  }
}
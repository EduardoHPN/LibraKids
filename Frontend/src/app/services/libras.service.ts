import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { API_URL } from './api.config';

// Formato dos payloads: docs/API_CONTRACT.md (na raiz do repositório).

export type Point = number[];

/** Mão do sinalizador. */
export type Handedness = 'left' | 'right';

/** Landmarks brutos de um frame (null = parte não detectada). */
export interface SequenceFrame {
  pose: Point[] | null;        // 33 × [x, y, z, visibility]
  face: Point[] | null;        // 468 ou 478 × [x, y, z]
  left_hand: Point[] | null;   // 21 × [x, y, z] — mão esquerda do sinalizador
  right_hand: Point[] | null;  // 21 × [x, y, z]
}

export interface Prediction {
  word: string;
  confidence: number;
  rank: number;
}

export interface PredictSequenceResponse {
  predictions: Prediction[];
  top1_word: string;
  top1_confidence: number;
  is_confident: boolean;
  frames_received: number;
}

export interface LetterPredictionResponse {
  success: boolean;
  correct: boolean;
  predicted_letter: string;
  target_letter: string;
  box?: [number, number, number, number];  // normalizado (0–1)
  handedness?: Handedness;
  confidence?: number;
  message: string;
}

export interface ClassesResponse {
  classes: string[];
  num_classes: number;
}

@Injectable({
  providedIn: 'root'
})
export class LibrasService {
  private apiUrl = API_URL;

  constructor(private http: HttpClient) { }

  /** Classifica uma sequência inteira de frames (10 a 60) — stateless. */
  predictSequence(frames: SequenceFrame[]): Observable<PredictSequenceResponse> {
    return this.http.post<PredictSequenceResponse>(`${this.apiUrl}/predict_sequence`, { frames });
  }

  getClasses(): Observable<ClassesResponse> {
    return this.http.get<ClassesResponse>(`${this.apiUrl}/classes`);
  }

  /** Classifica a letra a partir dos 21 landmarks [x, y] da mão e de qual mão é. */
  predictLetterLandmarks(
    hand: Point[] | null, targetLetter: string, handedness: Handedness | null,
  ): Observable<LetterPredictionResponse> {
    return this.http.post<LetterPredictionResponse>(
      `${this.apiUrl}/predict_letter_landmarks`,
      { hand, target_letter: targetLetter, handedness }
    );
  }

  /** Coleta de dataset — só funciona com LIBRAS_ENABLE_RECORD_DATA=1 no backend. */
  recordData(imageData: string, label: string): Observable<any> {
    const data = { image: imageData, label: label };
    return this.http.post(`${this.apiUrl}/record_data`, data);
  }
}

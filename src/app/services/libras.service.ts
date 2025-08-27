import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class LibrasService {
  private apiUrl = `http://localhost:5000`;

  constructor(private http: HttpClient) { }

  predictLetter(imageData: string, targetLetter: string): Observable<any> {
    const data = {
      image: imageData,
      target_letter: targetLetter
    };
    return this.http.post(`${this.apiUrl}/predict_letter`, data, { withCredentials: true });
  }

  recordData(imageData: string, label: string): Observable<any> {
    const data = {
      image: imageData,
      label: label
    };
    return this.http.post(`${this.apiUrl}/record_data`, data, { withCredentials: true });
  }
}
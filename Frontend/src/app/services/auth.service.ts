import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { Observable, tap } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private baseUrl = 'http://localhost:5001/';

  constructor(
    private http: HttpClient,
    private router: Router
  ) { }

  // --- Rotas de Autenticação ---
  register(credentials: { username: string, password: string }): Observable<any> {
    return this.http.post(`${this.baseUrl}/register`, credentials);
  }

  login(credentials: any): Observable<any> {
      return this.http.post<{ access_token: string }>(`${this.baseUrl}/login`, credentials).pipe(
        tap(response => {
          if (response.access_token) {
            localStorage.setItem('session_token', response.access_token);
          }
        })
      );
  }

// --- Rotas de IA ---
  // (predição de letras/palavras fica no LibrasService, com landmarks)
  recordData(data: any): Observable<any> {
    return this.http.post(`${this.baseUrl}/record_data`, data, { withCredentials: true });
  }

  // --- Rotas Protegidas de Conteúdo ---
  getProtectedPage(route: string): Observable<any> {
    return this.http.get(`${this.baseUrl}${route}`, { withCredentials: true });
  }

  logout(): void {
    localStorage.removeItem('session_token');
    this.router.navigate(['/login']);
    console.log('Utilizador deslogado.');
  }

  getToken(): string | null {
    return localStorage.getItem('session_token');
  }

  isLoggedIn(): boolean {
    return this.getToken() !== null;
  }
}
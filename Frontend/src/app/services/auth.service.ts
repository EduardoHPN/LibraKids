import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { Observable, tap } from 'rxjs';

import { API_URL } from './api.config';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private baseUrl = API_URL;

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
    return this.http.post(`${this.baseUrl}/record_data`, data);
  }

  // --- Rotas Protegidas de Conteúdo ---
  getProtectedPage(route: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/${route}`);
  }

  /**
   * Invalida o token no servidor e encerra a sessão local na hora, sem esperar
   * a resposta. O `subscribe` vem antes porque o interceptor lê o token ao
   * enviar. Se a chamada falhar, o token vale até expirar.
   */
  logout(): void {
    this.http.post(`${this.baseUrl}/logout`, {}).subscribe({ error: () => undefined });
    this.clearSession();
    console.log('Utilizador deslogado.');
  }

  /** Esquece o token e volta ao login. */
  clearSession(): void {
    localStorage.removeItem('session_token');
    this.router.navigate(['/login']);
  }

  getToken(): string | null {
    return localStorage.getItem('session_token');
  }

  isLoggedIn(): boolean {
    return this.getToken() !== null;
  }
}
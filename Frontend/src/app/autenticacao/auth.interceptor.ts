import { HttpErrorResponse, HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { catchError, throwError } from 'rxjs';

import { API_URL } from '../services/api.config';
import { AuthService } from '../services/auth.service';

// O 401 de /login significa senha errada, não sessão encerrada.
const ROUTES_WITHOUT_TOKEN = [`${API_URL}/login`, `${API_URL}/register`];

/** Envia o token às rotas da API e, se ele não valer mais, volta ao login. */
export const authInterceptor: HttpInterceptorFn = (req, next) => {
  if (!req.url.startsWith(`${API_URL}/`) || ROUTES_WITHOUT_TOKEN.includes(req.url)) {
    return next(req);
  }

  const auth = inject(AuthService);
  const token = auth.getToken();
  if (!token) {
    return next(req);
  }

  const authorized = req.clone({ setHeaders: { Authorization: `Bearer ${token}` } });
  return next(authorized).pipe(
    catchError((error: unknown) => {
      // Uma resposta atrasada do token antigo não pode derrubar um login mais novo.
      if (error instanceof HttpErrorResponse && error.status === 401 && auth.getToken() === token) {
        auth.clearSession();
      }
      return throwError(() => error);
    }),
  );
};

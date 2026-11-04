import { HttpClient, provideHttpClient, withInterceptors } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { Router } from '@angular/router';

import { API_URL } from '../services/api.config';
import { AuthService } from '../services/auth.service';
import { authInterceptor } from './auth.interceptor';

const TOKEN_KEY = 'session_token';
const PREDICT_URL = `${API_URL}/predict_sequence`;

describe('authInterceptor', () => {
  let http: HttpClient;
  let backend: HttpTestingController;
  let auth: AuthService;
  let navigate: jasmine.Spy;

  beforeEach(() => {
    localStorage.removeItem(TOKEN_KEY);
    navigate = jasmine.createSpy('navigate');
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(withInterceptors([authInterceptor])),
        provideHttpClientTesting(),
        { provide: Router, useValue: { navigate } },
      ],
    });
    http = TestBed.inject(HttpClient);
    backend = TestBed.inject(HttpTestingController);
    auth = TestBed.inject(AuthService);
  });

  afterEach(() => {
    backend.verify();
    localStorage.removeItem(TOKEN_KEY);
  });

  function post(url: string): void {
    http.post(url, {}).subscribe({ error: () => undefined });
  }

  it('envia o token às rotas da API', () => {
    localStorage.setItem(TOKEN_KEY, 'abc');
    post(PREDICT_URL);
    const req = backend.expectOne(PREDICT_URL);
    expect(req.request.headers.get('Authorization')).toBe('Bearer abc');
    req.flush({});
  });

  it('não envia o header sem token salvo', () => {
    post(PREDICT_URL);
    const req = backend.expectOne(PREDICT_URL);
    expect(req.request.headers.has('Authorization')).toBeFalse();
    req.flush({});
  });

  for (const url of [
    'assets/data/signs.json',
    'https://exemplo.com/predict_sequence',
    `${API_URL}.exemplo.com/predict_sequence`,
    `${API_URL}/login`,
    `${API_URL}/register`,
  ]) {
    it(`não envia o token para ${url}`, () => {
      localStorage.setItem(TOKEN_KEY, 'abc');
      post(url);
      const req = backend.expectOne(url);
      expect(req.request.headers.has('Authorization')).toBeFalse();
      req.flush({});
    });
  }

  it('com 401 da API, limpa o token, volta ao login e repassa o erro', () => {
    localStorage.setItem(TOKEN_KEY, 'abc');
    let status = 0;
    http.post(PREDICT_URL, {}).subscribe({ error: (e) => (status = e.status) });
    backend.expectOne(PREDICT_URL).flush({ message: 'x' }, { status: 401, statusText: 'Unauthorized' });
    expect(auth.getToken()).toBeNull();
    expect(navigate).toHaveBeenCalledOnceWith(['/login']);
    expect(status).toBe(401);
  });

  it('com 401 de /login (senha errada), mantém o token e não navega', () => {
    localStorage.setItem(TOKEN_KEY, 'abc');
    let message = '';
    http.post(`${API_URL}/login`, {}).subscribe({ error: (e) => (message = e.error.message) });
    backend
      .expectOne(`${API_URL}/login`)
      .flush({ message: 'Usuario ou senha incorretos.' }, { status: 401, statusText: 'Unauthorized' });
    expect(auth.getToken()).toBe('abc');
    expect(navigate).not.toHaveBeenCalled();
    expect(message).toBe('Usuario ou senha incorretos.');
  });

  it('ignora o 401 atrasado de um token antigo depois de um novo login', () => {
    localStorage.setItem(TOKEN_KEY, 'antigo');
    post(PREDICT_URL);
    localStorage.setItem(TOKEN_KEY, 'novo');
    backend.expectOne(PREDICT_URL).flush({}, { status: 401, statusText: 'Unauthorized' });
    expect(auth.getToken()).toBe('novo');
    expect(navigate).not.toHaveBeenCalled();
  });

  it('vários 401 seguidos do mesmo token navegam uma vez só', () => {
    localStorage.setItem(TOKEN_KEY, 'abc');
    post(PREDICT_URL);
    post(PREDICT_URL);
    for (const req of backend.match(PREDICT_URL)) {
      req.flush({}, { status: 401, statusText: 'Unauthorized' });
    }
    expect(navigate).toHaveBeenCalledTimes(1);
  });

  for (const status of [0, 500, 503]) {
    it(`mantém a sessão com status ${status} (servidor ou banco fora do ar)`, () => {
      localStorage.setItem(TOKEN_KEY, 'abc');
      post(PREDICT_URL);
      backend.expectOne(PREDICT_URL).flush({}, { status, statusText: 'Erro' });
      expect(auth.getToken()).toBe('abc');
      expect(navigate).not.toHaveBeenCalled();
    });
  }

  describe('AuthService.logout', () => {
    it('avisa o servidor com o token e encerra a sessão local na hora', () => {
      localStorage.setItem(TOKEN_KEY, 'abc');
      auth.logout();
      expect(auth.getToken()).toBeNull();
      expect(navigate).toHaveBeenCalledOnceWith(['/login']);

      const req = backend.expectOne(`${API_URL}/logout`);
      expect(req.request.method).toBe('POST');
      expect(req.request.headers.get('Authorization')).toBe('Bearer abc');
      req.flush({});
    });

    it('encerra a sessão local mesmo se o servidor falhar', () => {
      localStorage.setItem(TOKEN_KEY, 'abc');
      auth.logout();
      backend.expectOne(`${API_URL}/logout`).flush({}, { status: 500, statusText: 'Erro' });
      expect(auth.getToken()).toBeNull();
      expect(navigate).toHaveBeenCalledTimes(1);
    });
  });
});

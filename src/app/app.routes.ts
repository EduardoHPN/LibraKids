import { Routes } from '@angular/router';
import { Login } from './autenticacao/login/login';
import { Cadastro } from './autenticacao/cadastro/cadastro';

export const routes: Routes = [
  // Redireciona a rota raiz para '/login'
  { path: '', redirectTo: 'login', pathMatch: 'full' },
  
  // Rota para a tela de Login
  { path: 'login', component: Login },
  
  // Rota para a tela de Cadastro
  { path: 'cadastro', component: Cadastro },

  { path: 'home', loadComponent: () => import('./layout/main-layout/main-layout').then(m => m.MainLayoutComponent) },
];
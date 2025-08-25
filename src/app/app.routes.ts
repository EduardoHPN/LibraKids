// src/app/app.routes.ts

import { Routes } from '@angular/router';
import { Login } from './autenticacao/login/login';
import { Cadastro } from './autenticacao/cadastro/cadastro';
import { MainLayoutComponent } from './layout/main-layout/main-layout';
import { LearningPathComponent } from './layout/learning-path/learning-path';
import { DicionarioComponent } from './layout/dicionario/dicionario';
import { Profile } from './layout/profile/profile';
import { AlphabetPracticeComponent } from './layout/module/alphabet-practice.component/alphabet-practice.component';
import { DataCollectorComponent } from './machine-learning/data-collector/data-collector';
import { ModelProcessorComponent } from './machine-learning/model-processor.component/model-processor.component';
import { authGuard } from './autenticacao/auth.guard';


export const routes: Routes = [
  { path: '', redirectTo: 'login', pathMatch: 'full' },
  { path: 'login', component: Login },
  { path: 'cadastro', component: Cadastro },
  {
    path: 'app',
    component: MainLayoutComponent,
    canActivate: [authGuard],
    children: [
      { path: '', redirectTo: 'trilha', pathMatch: 'full' },
      { path: 'trilha', component: LearningPathComponent },
      { path: 'alphabet-practice', component: AlphabetPracticeComponent },
      { path: 'dicionario', component: DicionarioComponent },
      { path: 'data-collector', component: DataCollectorComponent },
      { path: 'processador', component: ModelProcessorComponent },
      { path: 'perfil', component: Profile }, 
    ]
  },
];
import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatDividerModule } from '@angular/material/divider';

// A interface do Módulo permanece a mesma
interface Module {
  title: string;
  description: string;
  // A "lição" agora representa uma funcionalidade interativa
  lesson: {
    route: string; // A rota para a nova funcionalidade
    title: string;
    icon: string;
  };
}

@Component({
  selector: 'app-learning-path',
  standalone: true,
  imports: [
    CommonModule,
    MatIconModule,
    MatButtonModule,
    MatCardModule,
    MatDividerModule
  ],
  templateUrl: './learning-path.html',
  styleUrls: ['./learning-path.scss']
})
export class LearningPathComponent {
  // --- NOVA ESTRUTURA DOS MÓDULOS ---
  learningPath: Module[] = [
    {
      title: 'Módulo 1: Prática do Alfabeto',
      description: 'Use sua câmera para treinar o alfabeto manual em tempo real. Nosso sistema identificará o sinal que você está fazendo.',
      lesson: {
        route: '/app/alphabet-practice', // Rota para o novo componente de câmera
        title: 'Iniciar Prática com Câmera',
        icon: 'camera_alt'
      }
    },
    {
      title: 'Módulo 2: Vocabulário Essencial',
      description: 'Aprenda palavras importantes do dia a dia, como saudações, alimentos e lugares, através de vídeos e exemplos práticos.',
      lesson: {
        route: '/app/word-practice', // Rota para o futuro módulo de palavras
        title: 'Aprender Palavras',
        icon: 'text_fields'
      }
    },
    {
      title: 'Módulo 3: Formação de Frases',
      description: 'Combine as palavras que você aprendeu para formar frases completas e entender a estrutura da Libras.',
      lesson: {
        route: '/app/phrase-practice', // Rota para o futuro módulo de frases
        title: 'Praticar Frases',
        icon: 'chat'
      }
    }
  ];

  constructor(private router: Router) {}

  // A função agora navega para a rota específica do módulo
  selectLesson(route: string): void {
    // Futuramente, você precisará criar os componentes e configurar estas rotas
    // em seu arquivo app.routes.ts
    this.router.navigate([route]);
    console.log(`Navegando para: ${route}`);
  }
}
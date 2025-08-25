import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { CommonModule } from '@angular/common';

// Imports do Angular Material
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'app-cadastro',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    RouterLink,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule
  ],
  templateUrl: './cadastro.html',
  styleUrls: ['./cadastro.scss']
})
export class Cadastro implements OnInit {
  // Declara o formulário e a variável para controlar a visibilidade da senha
  cadastroForm!: FormGroup;
  hidePassword = true;

  // Injeta o FormBuilder para criar o formulário
  constructor(private fb: FormBuilder) {}

  ngOnInit(): void {
    // Inicializa o formulário na inicialização do componente
    this.cadastroForm = this.fb.group({
      // Define os campos e suas validações
      nome: ['', [Validators.required]],
      email: ['', [Validators.required, Validators.email]],
      senha: ['', [Validators.required, Validators.minLength(6)]]
    });
  }

  // Função para alternar a visibilidade da senha
  togglePasswordVisibility(): void {
    this.hidePassword = !this.hidePassword;
  }

  // Função chamada ao enviar o formulário
  onSubmit(): void {
    if (this.cadastroForm.valid) {
      console.log('Dados do formulário de cadastro:', this.cadastroForm.value);
      // Aqui você chamaria um serviço para enviar os dados para o backend
    }
  }
}
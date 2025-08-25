import { Component, OnInit } from '@angular/core';
import { CommonModule, DatePipe } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatListModule } from '@angular/material/list';
import { MatDividerModule } from '@angular/material/divider';
import { MatDialog, MatDialogModule } from '@angular/material/dialog';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { Observable } from 'rxjs';
import { CourseService, UserData, UserStats } from '../../services/course';
import { EditProfileDialogComponent } from './edit-profile/edit-profile';

@Component({
  selector: 'app-perfil',
  standalone: true,
  imports: [
    CommonModule, DatePipe, MatCardModule, MatButtonModule, MatIconModule,
    MatListModule, MatDividerModule, MatDialogModule, MatSnackBarModule
  ],
  templateUrl: './profile.html',
  styleUrls: ['./profile.scss']
})
export class Profile implements OnInit {
  user$!: Observable<UserData>;
  stats$!: Observable<UserStats>;

  constructor(
    public dialog: MatDialog,
    private snackBar: MatSnackBar,
    private courseService: CourseService
  ) { }

  ngOnInit(): void {
    this.user$ = this.courseService.getUserData();
    this.stats$ = this.courseService.getUserStats();
  }

  editProfile(user: UserData): void {
    const dialogRef = this.dialog.open(EditProfileDialogComponent, {
      width: '450px',
      data: { ...user } // Passa uma cópia dos dados atuais para o modal
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result) {
        // Numa aplicação real, você enviaria 'result' para o backend para salvar.
        // Aqui, apenas exibimos uma notificação e atualizamos a visualização localmente.
        this.snackBar.open('Perfil atualizado com sucesso!', 'Fechar', {
          duration: 3000,
          verticalPosition: 'top',
          horizontalPosition: 'end'
        });
        
        // Cria um novo Observable com os dados atualizados para a view reagir
        this.user$ = new Observable(observer => observer.next({ ...user, ...result }));
      }
    });
  }
}
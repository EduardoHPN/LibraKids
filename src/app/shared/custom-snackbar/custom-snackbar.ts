import { Component, Inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MAT_SNACK_BAR_DATA, MatSnackBarRef } from '@angular/material/snack-bar';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'app-custom-snackbar',
  standalone: true,
  imports: [CommonModule, MatButtonModule, MatIconModule],
  templateUrl: './custom-snackbar.html',
  styleUrls: ['./custom-snackbar.scss']
})
export class CustomSnackbarComponent {
  constructor(
    public snackBarRef: MatSnackBarRef<CustomSnackbarComponent>,
    // A interface de dados agora é mais simples
    @Inject(MAT_SNACK_BAR_DATA) public data: { message: string; type: 'success' | 'error' }
  ) {}

  get icon(): string {
    return this.data.type === 'success' ? 'check_circle' : 'error';
  }

  close(): void {
    this.snackBarRef.dismiss();
  }
}
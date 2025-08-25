import { Injectable } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { CustomSnackbarComponent } from '../shared/custom-snackbar/custom-snackbar';

@Injectable({
  providedIn: 'root'
})
export class NotificationService {

  constructor(private snackBar: MatSnackBar) { }

  showSuccess(message: string): void {
    this.snackBar.openFromComponent(CustomSnackbarComponent, {
      data: {
        message: message,
        // Sem título
        type: 'success'
      },
      duration: 4000,
      horizontalPosition: 'end',
      verticalPosition: 'top',
      panelClass: ['custom-snackbar-panel']
    });
  }

  showError(message: string): void {
    this.snackBar.openFromComponent(CustomSnackbarComponent, {
      data: {
        message: message,
        // Sem título
        type: 'error'
      },
      duration: 5000,
      horizontalPosition: 'end',
      verticalPosition: 'top',
      panelClass: ['custom-snackbar-panel']
    });
  }
}
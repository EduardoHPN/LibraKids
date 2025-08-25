import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map } from 'rxjs';

// --- Interfaces Globais ---
export interface SignContent {
  letter: string;
  imageUrl: string;
  description: string;
}

export interface LessonContent {
  id: number;
  title: string;
  objective: string;
  content: SignContent[];
}

export interface Module {
  title: string;
  description: string;
  lessons: { id: number; title: string; icon: string; }[];
}

export interface UserData {
  name: string;
  email: string;
  joinDate: Date;
  profileImageUrl: string;
}

export interface UserStats {
  modulesCompleted: number;
  totalModules: number;
  signsLearned: number;
  totalSigns: number;
}

@Injectable({
  providedIn: 'root'
})
export class CourseService {
  constructor(private http: HttpClient) { }

  getAllSigns(): Observable<SignContent[]> {
    return this.http.get<SignContent[]>('assets/data/signs.json').pipe(
      map(signs => signs.sort((a, b) => a.letter.localeCompare(b.letter)))
    );
  }

  // --- MÉTODOS PARA O PERFIL (COM DADOS DE EXEMPLO) ---

  getUserData(): Observable<UserData> {
    const user: UserData = {
      name: 'Alex Green',
      email: 'alex.green@email.com',
      joinDate: new Date('2025-07-15T10:00:00Z'),
      profileImageUrl: 'assets/images/avatar.png'
    };
    return new Observable(observer => observer.next(user));
  }

  getUserStats(): Observable<UserStats> {
    const stats: UserStats = {
      modulesCompleted: 1,
      totalModules: 4,
      signsLearned: 7,
      totalSigns: 26
    };
    return new Observable(observer => observer.next(stats));
  }
}
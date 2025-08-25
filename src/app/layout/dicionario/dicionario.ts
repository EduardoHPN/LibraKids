import { Component, OnInit } from '@angular/core';
import { CommonModule, KeyValue } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatIconModule } from '@angular/material/icon';
import { MatExpansionModule } from '@angular/material/expansion';
import { Observable, BehaviorSubject, combineLatest } from 'rxjs';
import { map, startWith } from 'rxjs/operators';
import { CourseService, SignContent } from '../../services/course';

interface GroupedSigns { [key: string]: SignContent[]; }

@Component({
  selector: 'app-dicionario',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatIconModule,
    MatExpansionModule
  ],
  templateUrl: './dicionario.html',
  styleUrls: ['./dicionario.scss']
})
export class DicionarioComponent implements OnInit {
  private searchTerm = new BehaviorSubject<string>('');
  groupedSigns$!: Observable<GroupedSigns>;
  resultsCount$!: Observable<number>;

  constructor(private courseService: CourseService) {}

  ngOnInit(): void {
    const allSigns$ = this.courseService.getAllSigns();

    const filteredSigns$ = combineLatest([ allSigns$, this.searchTerm.pipe(startWith('')) ]).pipe(
      map(([signs, term]) => {
        const searchTerm = term.toLowerCase().trim();
        if (!searchTerm) return signs;
        return signs.filter(sign =>
          sign.letter.toLowerCase() === searchTerm
        );
      })
    );

    this.groupedSigns$ = filteredSigns$.pipe(
      map(signs => signs.reduce((acc, sign) => {
        const letter = sign.letter.toUpperCase();
        if (!acc[letter]) acc[letter] = [];
        acc[letter].push(sign);
        return acc;
      }, {} as GroupedSigns))
    );

    this.resultsCount$ = filteredSigns$.pipe(map(signs => signs.length));
  }

  onSearchChange(term: string): void {
    this.searchTerm.next(term);
  }

  // Mantém a ordem alfabética dos grupos no template
  originalOrder = (a: KeyValue<string, SignContent[]>, b: KeyValue<string, SignContent[]>): number => a.key.localeCompare(b.key);
}
import numpy as np
from typing import Tuple, Optional, List
import time
from functools import lru_cache

class Gomoku:
    def __init__(self, size: int = 20, win_length: int = 5, difficulty: int = 60):
        self.size = size  # Размер доски
        self.win_length = win_length  # Длина последовательности для победы
        self.board = np.zeros((size, size), dtype=np.int8)  # Создание пустой доски
        self.current_player = 1  # Игрок, который делает ход (1 или -1)
        self.move_count = 0  # Количество сделанных ходов
        self.difficulty = max(1, min(10, difficulty))  # Установка сложности игры
        self.last_move = None  # Последний сделанный ход

        # Генерация списка всех индексов на доске
        self.board_indices = [(i, j) for i in range(size) for j in range(size)]
        self.directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Направления для проверки последовательностей

        # Весовые коэффициенты для оценки позиций на доске
        self.weights = {
            5: 1_000_000,
            4: 100_000,
            '4b': 10_000,
            3: 1_000,
            '3b': 100,
            2: 10,
            '2b': 1
        }

    def is_valid_coord(self, x: int, y: int) -> bool:
        # Проверка, находятся ли координаты в пределах доски
        return 0 <= x < self.size and 0 <= y < self.size

    @lru_cache(maxsize=1024)
    def check_sequence(self, x: int, y: int, dx: int, dy: int, player: int) -> Tuple[int, bool]:
        # Проверка последовательности одинаковых фишек в заданном направлении
        count = 1  # Количество фишек в последовательности
        blocked_ends = 0  # Количество заблокированных концов последовательности

        # Проверка вперед по направлению
        x1, y1 = x + dx, y + dy
        while self.is_valid_coord(x1, y1) and self.board[x1][y1] == player:
            count += 1
            x1, y1 = x1 + dx, y1 + dy
        if not (self.is_valid_coord(x1, y1) and self.board[x1][y1] == 0):
            blocked_ends += 1

        # Проверка назад по направлению
        x1, y1 = x - dx, y - dy
        while self.is_valid_coord(x1, y1) and self.board[x1][y1] == player:
            count += 1
            x1, y1 = x1 - dx, y1 - dy
        if not (self.is_valid_coord(x1, y1) and self.board[x1][y1] == 0):
            blocked_ends += 1

        return count, blocked_ends < 2  # Возвращение количества фишек и информации о заблокированных концах

    def evaluate_position(self, x: int, y: int, player: int) -> int:
        # Оценка позиции для заданного игрока
        if not self.is_valid_coord(x, y):
            return 0

        score = 0  # Начальное значение оценки
        for dx, dy in self.directions:
            length, is_open = self.check_sequence(x, y, dx, dy, player)

            if length >= self.win_length:
                return self.weights[5]  # Ранний выход для выигрышной позиции
            elif length == 4:
                score += self.weights[4] if is_open else self.weights['4b']
            elif length == 3:
                score += self.weights[3] if is_open else self.weights['3b']
            elif length == 2:
                score += self.weights[2] if is_open else self.weights['2b']

        return score  # Возвращение окончательной оценки позиции

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        # Получение списка допустимых ходов
        if self.move_count == 0:
            return [(self.size // 2, self.size // 2)]  # Первый ход в центр доски

        # Быстрая проверка на возможность выигрыша или блокировки противника
        for player in [self.current_player, -self.current_player]:
            for i, j in self.get_nearby_empty_cells():
                if self.would_win(i, j, player):
                    return [(i, j)]

        moves = []
        seen = set()

        for i, j in self.get_nearby_empty_cells():
            if (i, j) not in seen:
                attack_score = self.evaluate_position(i, j, self.current_player)
                defense_score = self.evaluate_position(i, j, -self.current_player)
                score = max(attack_score, defense_score)
                moves.append((score, (i, j)))
                seen.add((i, j))

        moves.sort(reverse=True)
        return [move for _, move in moves[:max(5, self.difficulty * 2)]]

    def get_nearby_empty_cells(self) -> List[Tuple[int, int]]:
        # Получение списка пустых клеток рядом с занятыми
        nearby = set()
        for i, j in self.board_indices:
            if self.board[i][j] != 0:
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni, nj = i + di, j + dj
                        if self.is_valid_coord(ni, nj) and self.board[ni][nj] == 0:
                            nearby.add((ni, nj))
        return list(nearby)

    def would_win(self, x: int, y: int, player: int) -> bool:
        # Проверка, приведет ли ход к победе
        self.board[x][y] = player
        is_win = any(self.check_sequence(x, y, dx, dy, player)[0] >= self.win_length 
                    for dx, dy in self.directions)
        self.board[x][y] = 0
        return is_win

    def make_move(self, x: int, y: int) -> bool:
        # Совершение хода
        if self.is_valid_coord(x, y) and self.board[x][y] == 0:
            self.board[x][y] = self.current_player
            self.move_count += 1
            self.last_move = (x, y)
            self.check_sequence.cache_clear()
            return True
        return False

    def get_ai_move(self) -> Tuple[int, int]:
        # Получение хода для ИИ
        valid_moves = self.get_valid_moves()
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        depth = max(2, min(self.difficulty // 2, 6))  # Увеличение глубины поиска до 6
        _, move = self.minimax(depth, float('-inf'), float('inf'), True)
        return move if move else valid_moves[0]

    def minimax(self, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[int, Optional[Tuple[int, int]]]:
        # Алгоритм minimax с альфа-бета отсечением
        if depth == 0 or self.is_winner(self.current_player) or self.is_winner(-self.current_player):
            return self.evaluate_board(), None

        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return 0, None

        best_move = valid_moves[0]
        if maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                self.board[move[0]][move[1]] = self.current_player
                self.current_player *= -1
                eval_score, _ = self.minimax(depth - 1, alpha, beta, False)
                self.current_player *= -1
                self.board[move[0]][move[1]] = 0
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in valid_moves:
                self.board[move[0]][move[1]] = self.current_player
                self.current_player *= -1
                eval_score, _ = self.minimax(depth - 1, alpha, beta, True)
                self.current_player *= -1
                self.board[move[0]][move[1]] = 0
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def evaluate_board(self) -> int:
        # Оценка текущего состояния доски
        if self.last_move is None:
            return 0
            
        x, y = self.last_move
        score = 0
        for i in range(max(0, x-2), min(self.size, x+3)):
            for j in range(max(0, y-2), min(self.size, y+3)):
                if self.board[i][j] != 0:
                    if self.board[i][j] == self.current_player:
                        score += self.evaluate_position(i, j, self.current_player)
                    else:
                        score -= self.evaluate_position(i, j, -self.current_player)
        return score

    def is_winner(self, player: int) -> bool:
        # Проверка, выиграл ли игрок
        if self.last_move is None:
            return False
        x, y = self.last_move
        return any(self.check_sequence(x, y, dx, dy, player)[0] >= self.win_length 
                  for dx, dy in self.directions)

    def print_board(self):
        # Печать доски
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print('   ', end='')
        for i in range(self.size):
            print(f'{i:2}', end=' ')
        print()
        
        for i in range(self.size):
            print(f'{i:2} ', end='')
            for j in range(self.size):
                print(f' {symbols[self.board[i][j]]}', end=' ')
            print()

def play_game():
    # Функция для игры в Гомоку
    print("Выберите сложность игры:")
    difficulty = int(input("Введите число от 1 до 10: "))
    while not 1 <= difficulty <= 10:
        print("Неверная сложность! Выберите число от 1 до 10.")
        difficulty = int(input("Выберите сложность от 1 до 10: "))
        
    game = Gomoku(difficulty=difficulty)
    
    mode = input("\nВыберите режим:\n1 - Игрок ходит первым\n2 - Компьютер ходит первым\nВведите 1 или 2: ")
    while mode not in ['1', '2']:
        print("Некорректный выбор! Пожалуйста, выберите 1 или 2.")
        mode = input("Выберите режим:\n1 - Игрок ходит первым\n2 - Компьютер ходит первым\nВведите 1 или 2: ")
    
    if mode == '2':
        game.current_player = -1
    
    while True:
        game.print_board()
        
        if game.current_player == 1:
            try:
                row = int(input("\nВведите номер строки: "))
                col = int(input("Введите номер столбца: "))
                if not game.make_move(row, col):
                    print("Недопустимый ход! Попробуйте снова.")
                    continue
            except ValueError:
                print("Пожалуйста, введите числа!")
                continue
        else:
            print("\nКомпьютер думает...")
            start_time = time.time()
            row, col = game.get_ai_move()
            think_time = time.time() - start_time
            game.make_move(row, col)
            print(f"Компьютер походил: {row}, {col} (время: {think_time:.2f}с)")
        
        if game.is_winner(game.current_player):
            game.print_board()
            winner = "Игрок" if game.current_player == 1 else "Компьютер"
            print(f"\n{winner} победил!")
            break
        
        if game.move_count == game.size * game.size:
            game.print_board()
            print("\nНичья!")
            break
        
        game.current_player *= -1

if __name__ == "__main__":
    play_game()

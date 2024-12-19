import numpy as np
from typing import Tuple, Optional, List
import time

class Gomoku:
    def __init__(self, size: int = 20, win_length: int = 5, difficulty: int = 5):
        self.size = size
        self.win_length = win_length
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1
        self.move_count = 0
        self.difficulty = max(1, min(10, difficulty))
        
        # Улучшенные веса для разных паттернов
        self.weights = {
            5: 1000000000,     # Выигрышная комбинация
            '5b': 500000000,   # Блокированная пятерка
            4: 100000000,      # Открытая четверка
            '4b': 1000000,     # Блокированная четверка
            3: 100000,         # Открытая тройка
            '3b': 10000,       # Блокированная тройка
            2: 1000,           # Открытая двойка
            '2b': 100,         # Блокированная двойка
            'fork': 5000000    # Вилка (множественная угроза)
        }
        
        self.directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        self.pattern_cache = {}
        self.threat_cache = {}
        self.center_weight = 2.0  # Вес для центральных позиций
        self.history_table = {}   # Таблица истории ходов для улучшения альфа-бета отсечений

    def get_pattern_key(self, x: int, y: int, dx: int, dy: int, length: int) -> str:
        """Получает ключ паттерна для определенной последовательности."""
        pattern = []
        for i in range(-length, length + 1):
            nx, ny = x + i * dx, y + i * dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                pattern.append(str(self.board[nx][ny]))
            else:
                pattern.append('x')  # out of bounds
        return ''.join(pattern)

    def detect_fork(self, x: int, y: int, player: int) -> bool:
        """Определяет наличие вилки (множественной угрозы)."""
        threats = 0
        self.board[x][y] = player
        
        # Проверяем угрозы по всем направлениям
        for dx, dy in self.directions:
            pattern = self.get_pattern_key(x, y, dx, dy, 4)
            if pattern in self.threat_cache:
                threats += self.threat_cache[pattern]
            else:
                # Проверяем на открытую четверку или тройку
                length, is_open = self.check_sequence(x, y, dx, dy, player)
                if (length == 4 and is_open) or (length == 3 and is_open):
                    threats += 1
                self.threat_cache[pattern] = threats
        
        self.board[x][y] = 0
        return threats >= 2

    def check_sequence(self, x: int, y: int, dx: int, dy: int, player: int) -> Tuple[int, bool]:
        key = (x, y, dx, dy, player)
        if key in self.pattern_cache:
            return self.pattern_cache[key]

        count = 1
        blocked_ends = 0
        
        # Проверка в одном направлении
        x1, y1 = x + dx, y + dy
        while 0 <= x1 < self.size and 0 <= y1 < self.size and self.board[x1][y1] == player:
            count += 1
            x1 += dx
        if not (0 <= x1 < self.size and 0 <= y1 < self.size and self.board[x1][y1] == 0):
            blocked_ends += 1
            
        # Проверка в противоположном направлении
        x1, y1 = x - dx, y - dy
        while 0 <= x1 < self.size and 0 <= y1 < self.size and self.board[x1][y1] == player:
            count += 1
            x1 -= dx
        if not (0 <= x1 < self.size and 0 <= y1 < self.size and self.board[x1][y1] == 0):
            blocked_ends += 1
            
        result = (count, blocked_ends < 2)
        self.pattern_cache[key] = result
        return result

    def evaluate_position(self, x: int, y: int, player: int) -> int:
        if not (0 <= x < self.size and 0 <= y < self.size):
            return 0
            
        score = 0
        # Базовая оценка
        for dx, dy in self.directions:
            length, is_open = self.check_sequence(x, y, dx, dy, player)
            
            if length >= self.win_length:
                score += self.weights[5]
            elif length == 4:
                score += self.weights[4] if is_open else self.weights['4b']
            elif length == 3:
                score += self.weights[3] if is_open else self.weights['3b']
            elif length == 2:
                score += self.weights[2] if is_open else self.weights['2b']

        # Дополнительные факторы оценки
        if self.detect_fork(x, y, player):
            score += self.weights['fork']
            
        # Учет центральности позиции
        center_x, center_y = self.size // 2, self.size // 2
        distance_to_center = abs(x - center_x) + abs(y - center_y)
        center_bonus = (self.size - distance_to_center) * self.center_weight
        score += center_bonus
        
        # Учет связности с другими камнями
        connected_stones = self.count_connected_stones(x, y, player)
        score += connected_stones * 100

        return score

    def count_connected_stones(self, x: int, y: int, player: int) -> int:
        """Подсчитывает количество связанных камней в радиусе 2."""
        count = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx][ny] == player:
                    count += 1
        return count

    def evaluate_board(self) -> int:
        score = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != 0:
                    if self.board[i][j] == self.current_player:
                        score += self.evaluate_position(i, j, self.current_player)
                    else:
                        score -= self.evaluate_position(i, j, -self.current_player)
        return score

    def find_winning_move(self, player: int) -> Optional[Tuple[int, int]]:
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    self.board[i][j] = player
                    if self.is_winner(player):
                        self.board[i][j] = 0
                        return (i, j)
                    self.board[i][j] = 0
        return None

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        if self.move_count == 0:
            return [(self.size // 2, self.size // 2)]

        winning_move = self.find_winning_move(self.current_player)
        if winning_move:
            return [winning_move]

        blocking_move = self.find_winning_move(-self.current_player)
        if blocking_move:
            return [blocking_move]
        
        moves = []
        seen = set()
        
        # Расширенный радиус поиска вокруг существующих камней
        search_radius = 3 if self.move_count < 30 else 2
        
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != 0:
                    for di in range(-search_radius, search_radius + 1):
                        for dj in range(-search_radius, search_radius + 1):
                            ni, nj = i + di, j + dj
                            if (ni, nj) not in seen and 0 <= ni < self.size and 0 <= nj < self.size and self.board[ni][nj] == 0:
                                attack_score = self.evaluate_position(ni, nj, self.current_player)
                                defense_score = self.evaluate_position(ni, nj, -self.current_player)
                                history_score = self.history_table.get((ni, nj), 0)
                                score = max(attack_score, defense_score) + history_score
                                moves.append((score, (ni, nj)))
                                seen.add((ni, nj))
        
        moves.sort(reverse=True)
        num_moves = max(5, min(20, self.difficulty * 3))
        return [move for _, move in moves[:num_moves]]

    def get_search_depth(self) -> int:
        """Динамическая глубина поиска в зависимости от стадии игры."""
        empty_cells = self.size * self.size - self.move_count
        if empty_cells < 30:  # Эндшпиль
            return min(self.difficulty + 2, 7)
        elif empty_cells < 100:  # Мидгейм
            return min(self.difficulty + 1, 6)
        else:  # Начало игры
            return max(2, min(self.difficulty // 2, 5))

    def is_winner(self, player: int) -> bool:
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == player:
                    for dx, dy in self.directions:
                        count = 1
                        # Проверяем в одном направлении
                        x1, y1 = i + dx, j + dy
                        while 0 <= x1 < self.size and 0 <= y1 < self.size and self.board[x1][y1] == player:
                            count += 1
                            x1 += dx
                        # Проверяем в противоположном направлении
                        x1, y1 = i - dx, j - dy
                        while 0 <= x1 < self.size and 0 <= y1 < self.size and self.board[x1][y1] == player:
                            count += 1
                            x1 -= dx
                        if count >= self.win_length:
                            return True
        return False


    def make_move(self, x: int, y: int) -> bool:
        if 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == 0:
            self.board[x][y] = self.current_player
            self.move_count += 1
            self.pattern_cache.clear()
            self.threat_cache.clear()
            return True
        return False

    def minimax(self, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[int, Optional[Tuple[int, int]]]:
        if depth == 0:
            return self.evaluate_board(), None

        moves = self.get_valid_moves()
        moves.sort(key=lambda m: self.history_table.get((m[0], m[1]), 0), reverse=True)

        if maximizing:
            max_eval = float('-inf')
            best_move = None
            for move in moves:
                self.board[move[0]][move[1]] = self.current_player
                self.current_player *= -1
                eval_score, _ = self.minimax(depth - 1, alpha, beta, False)
                self.current_player *= -1
                self.board[move[0]][move[1]] = 0
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                    self.history_table[(move[0], move[1])] = self.history_table.get((move[0], move[1]), 0) + 2 ** depth
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in moves:
                self.board[move[0]][move[1]] = self.current_player
                self.current_player *= -1
                eval_score, _ = self.minimax(depth - 1, alpha, beta, True)
                self.current_player *= -1
                self.board[move[0]][move[1]] = 0
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                    self.history_table[(move[0], move[1])] = self.history_table.get((move[0], move[1]), 0) + 2 ** depth
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def get_ai_move(self) -> Tuple[int, int]:
        winning_move = self.find_winning_move(self.current_player)
        if winning_move:
            return winning_move
            
        blocking_move = self.find_winning_move(-self.current_player)
        if blocking_move:
            return blocking_move
        
        depth = self.get_search_depth()
        _, move = self.minimax(depth, float('-inf'), float('inf'), True)
        return move if move else self.get_valid_moves()[0]

    def print_board(self):
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
    difficulty = int(input("Выберите сложность от 1 до 10 (1 - очень легко, 10 - очень сложно): "))
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
                row = int(input("Введите номер строки: "))
                col = int(input("Введите номер столбца: "))
                if not game.make_move(row, col):
                    print("Недопустимый ход! Попробуйте снова.")
                    continue
            except ValueError:
                print("Пожалуйста, введите числа!")
                continue
        else:
            print("Компьютер думает...")
            start_time = time.time()
            row, col = game.get_ai_move()
            think_time = time.time() - start_time
            game.make_move(row, col)
            print(f"Компьютер походил: {row}, {col} (время: {think_time:.2f}с)")
        
        if game.is_winner(game.current_player):
            game.print_board()
            winner = "Игрок" if game.current_player == 1 else "Компьютер"
            print(f"{winner} победил!")
            break
        
        if game.move_count == game.size * game.size:
            game.print_board()
            print("Ничья!")
            break
        
        game.current_player *= -1


if __name__ == "__main__":
    play_game()

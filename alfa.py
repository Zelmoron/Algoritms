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
        
        self.weights = {
            5: 1000000000,    
            4: 100000000,     
            '4b': 1000000,    
            3: 100000,        
            '3b': 10000,      
            2: 1000,          
            '2b': 100         
        }
        
        self.directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        self.pattern_cache = {}

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

    def check_sequence(self, x: int, y: int, dx: int, dy: int, player: int) -> Tuple[int, bool]:
        key = (x, y, dx, dy, player)
        if key in self.pattern_cache:
            return self.pattern_cache[key]

        count = 1
        blocked_ends = 0
        

        x1, y1 = x + dx, y + dy
        while 0 <= x1 < self.size and 0 <= y1 < self.size and self.board[x1][y1] == player:
            count += 1
            x1 += dx
            y1 += dy
        if not (0 <= x1 < self.size and 0 <= y1 < self.size and self.board[x1][y1] == 0):
            blocked_ends += 1
            
       
        x1, y1 = x - dx, y - dy
        while 0 <= x1 < self.size and 0 <= y1 < self.size and self.board[x1][y1] == player:
            count += 1
            x1 -= dx
            y1 -= dy
        if not (0 <= x1 < self.size and 0 <= y1 < self.size and self.board[x1][y1] == 0):
            blocked_ends += 1
            
        result = (count, blocked_ends < 2)
        self.pattern_cache[key] = result
        return result

    def evaluate_position(self, x: int, y: int, player: int) -> int:
        if not (0 <= x < self.size and 0 <= y < self.size):
            return 0
            
        score = 0
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
                    
        return score

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
        
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != 0:
                    for di in range(-2, 3):
                        for dj in range(-2, 3):
                            ni, nj = i + di, j + dj
                            if (ni, nj) not in seen and 0 <= ni < self.size and 0 <= nj < self.size and self.board[ni][nj] == 0:
                                attack_score = self.evaluate_position(ni, nj, self.current_player)
                                defense_score = self.evaluate_position(ni, nj, -self.current_player)
                                score = max(attack_score, defense_score)
                                moves.append((score, (ni, nj)))
                                seen.add((ni, nj))
        
        moves.sort(reverse=True)
        return [move for _, move in moves[:max(5, self.difficulty * 2)]]

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

    def get_search_depth(self) -> int:
        return max(2, min(self.difficulty // 2, 5))

    def is_winner(self, player: int) -> bool:
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == player:
                    for dx, dy in self.directions:
                        count = 1
                        x1, y1 = i + dx, j + dy
                        while 0 <= x1 < self.size and 0 <= y1 < self.size and self.board[x1][y1] == player:
                            count += 1
                            if count >= self.win_length:
                                return True
                            x1 += dx
                            y1 += dy
                        
                        x1, y1 = i - dx, j - dy
                        while 0 <= x1 < self.size and 0 <= y1 < self.size and self.board[x1][y1] == player:
                            count += 1
                            if count >= self.win_length:
                                return True
                            x1 -= dx
                            y1 -= dy
        return False

    def make_move(self, x: int, y: int) -> bool:
        if 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == 0:
            self.board[x][y] = self.current_player
            self.move_count += 1
            self.pattern_cache.clear()
            return True
        return False

    def minimax(self, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[int, Optional[Tuple[int, int]]]:
        if depth == 0:
            return self.evaluate_board(), None

        if maximizing:
            max_eval = float('-inf')
            best_move = None
            for move in self.get_valid_moves():
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
            best_move = None
            for move in self.get_valid_moves():
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
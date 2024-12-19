from alfa2 import Ant
import numpy as np
from typing import List, Tuple

def aco_algorithm(
    num_ants: int,
    num_iterations: int,
    evaporation_rate: float,
    start_point: int,
    end_point: int,
    distance_matrix: np.ndarray
) -> Tuple[List[float], List[float], List[float], List[int], float]:
    """
    Алгоритм оптимизации муравьиных колоний (ACO).

    :param num_ants: Количество муравьёв, участвующих в каждой итерации.
    :param num_iterations: Общее количество итераций алгоритма.
    :param evaporation_rate: Коэффициент испарения феромонов (значение от 0 до 1).
    :param start_point: Индекс стартовой точки маршрута.
    :param end_point: Индекс конечной точки маршрута.
    :param distance_matrix: Матрица расстояний между точками.
    :return:
        - История средней длины маршрутов за итерацию (distances_history).
        - История средней концентрации феромонов на пути (pheromone_history).
        - История вероятностей переходов между узлами (probabilities_history).
        - Лучший маршрут (best_tour).
        - Длина лучшего маршрута (best_distance).
    """
    num_nodes: int = len(distance_matrix)  # Общее количество узлов
    pheromone_matrix: np.ndarray = np.ones((num_nodes, num_nodes))  # Матрица феромонов
    visibility_matrix: np.ndarray = np.zeros_like(distance_matrix, dtype=float)  # Матрица видимости (1 / расстояние)

    # Заполнение матрицы видимости
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and distance_matrix[i][j] > 0:
                visibility_matrix[i][j] = 1.0 / distance_matrix[i][j]  # Видимость обратно пропорциональна расстоянию

    # Инициализация переменных для хранения лучшего результата
    best_distance: float = float('inf')  # Наименьшая длина маршрута
    best_tour: List[int] = None          # Лучший маршрут
    distances_history: List[float] = [] # История средней длины маршрутов за итерацию
    pheromone_history: List[float] = [] # История средней концентрации феромонов
    probabilities_history: List[float] = []  # История средних вероятностей переходов между узлами

    # Основной цикл алгоритма
    for iteration in range(num_iterations):
        # Создание списка муравьёв для текущей итерации
        ants: List[Ant] = [Ant(num_nodes, start_point, end_point) for _ in range(num_ants)]
        iteration_distances: List[float] = []      # Список расстояний, пройденных муравьями в текущей итерации
        iteration_probabilities: List[float] = []  # Список вероятностей переходов в текущей итерации

        # Построение маршрутов для каждого муравья
        for ant in ants:
            # Муравей строит маршрут, используя текущие феромоны и матрицу видимости
            ant.construct_tour(pheromone_matrix, visibility_matrix, distance_matrix)
            iteration_distances.append(ant.distance)  # Сохраняем длину маршрута муравья
            iteration_probabilities.extend(ant.probabilities)  # Сохраняем вероятности переходов муравья

            # Обновление лучшего маршрута
            if ant.distance < best_distance:
                best_distance = ant.distance
                best_tour = ant.tour.copy()  # Копируем маршрут

        # Сохраняем среднюю длину маршрутов и вероятности за текущую итерацию
        distances_history.append(np.mean(iteration_distances))
        probabilities_history.append(np.mean(iteration_probabilities))

        # Испарение феромонов
        pheromone_matrix *= (1 - evaporation_rate)

        # Обновление феромонов по пройденным маршрутам
        for ant in ants:
            deposit: float = 1.0 / ant.distance if ant.distance > 0 else 0  # Размер "вклада" феромонов
            for i in range(len(ant.tour) - 1):
                current: int = ant.tour[i]
                next_city: int = ant.tour[i + 1]
                pheromone_matrix[current][next_city] += deposit

        # Сохранение средней концентрации феромонов на лучшем пути
        if best_tour:
            path_pheromones: List[float] = [
                pheromone_matrix[best_tour[i]][best_tour[i + 1]] for i in range(len(best_tour) - 1)
            ]
            pheromone_history.append(np.mean(path_pheromones))

    return distances_history, pheromone_history, probabilities_history, best_tour, best_distance

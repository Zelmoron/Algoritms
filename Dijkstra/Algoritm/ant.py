import numpy as np
from typing import List, Set

class Ant:
    def __init__(
        self,
        num_nodes: int,
        start_point: int,
        end_point: int,
        alpha: float = 1.0,
        beta: float = 2.0
    ) -> None:
        """
        Инициализация объекта муравья.

        :param num_nodes: Количество узлов в графе.
        :param start_point: Стартовая точка маршрута.
        :param end_point: Конечная точка маршрута.
        :param alpha: Влияние феромонов на выбор пути.
        :param beta: Влияние видимости на выбор пути.
        """
        self.num_nodes: int = num_nodes
        self.start_point: int = start_point
        self.end_point: int = end_point
        self.tour: List[int] = []  # Хранение маршрута, по которому прошёл муравей
        self.distance: float = 0.0  # Общая длина маршрута
        self.alpha: float = alpha  # Важность феромонов
        self.beta: float = beta  # Важность видимости
        self.probabilities: List[float] = []  # Список вероятностей выбора городов

    def construct_tour(
        self,
        pheromone_matrix: np.ndarray,
        visibility_matrix: np.ndarray,
        distance_matrix: np.ndarray
    ) -> None:
        """
        Построение маршрута для муравья.

        :param pheromone_matrix: Матрица феромонов.
        :param visibility_matrix: Матрица видимости (1 / расстояние).
        :param distance_matrix: Матрица расстояний.
        """
        self.tour = [self.start_point]  # Маршрут начинается со стартовой точки
        current: int = self.start_point  # Текущая точка
        available_nodes: Set[int] = set(range(self.num_nodes))  # Узлы, доступные для посещения
        available_nodes.remove(self.start_point)  # Исключаем стартовую точку

        while current != self.end_point:
            # Если нет доступных узлов и мы не в конечной точке, принудительно добавляем конечную точку
            if not available_nodes and current != self.end_point:
                self.tour.append(self.end_point)
                break

            # Формируем список доступных городов
            cities: List[int] = list(available_nodes) + [self.end_point] \
                if self.end_point not in self.tour else list(available_nodes)

            if not cities:  # Если больше нет доступных городов, завершаем маршрут
                break

            probabilities: List[float] = []  # Вероятности для выбора следующего города
            denominator: float = 0.0  # Сумма знаменателей для нормализации вероятностей

            # Рассчитываем знаменатель вероятностей
            for city in cities:
                tau: float = pheromone_matrix[current][city]  # Уровень феромонов
                eta: float = visibility_matrix[current][city]  # Видимость (1 / расстояние)
                denominator += (tau ** self.alpha) * (eta ** self.beta)

            # Рассчитываем вероятность для каждого доступного города
            for next_city in cities:
                tau: float = pheromone_matrix[current][next_city]
                eta: float = visibility_matrix[current][next_city]
                probability: float = (
                    ((tau ** self.alpha) * (eta ** self.beta)) / denominator
                    if denominator > 0
                    else 1.0 / len(cities)
                )
                probabilities.append(probability)

            # Выбираем следующий город на основе вероятностей
            next_city: int = np.random.choice(cities, p=probabilities)
            self.probabilities.append(max(probabilities))  # Сохраняем максимальную вероятность перехода
            self.tour.append(next_city)  # Добавляем выбранный город в маршрут
            current = next_city  # Переходим в выбранный город
            if next_city in available_nodes:
                available_nodes.remove(next_city)  # Убираем посещённый город из доступных

        # Рассчитываем длину маршрута после его построения
        self.distance = self.calculate_distance(distance_matrix)

    def calculate_distance(self, distance_matrix: np.ndarray) -> float:
        """
        Вычисление длины маршрута.

        :param distance_matrix: Матрица расстояний.
        :return: Общая длина маршрута.
        """
        return sum(
            distance_matrix[self.tour[i]][self.tour[i + 1]]
            for i in range(len(self.tour) - 1)
        )

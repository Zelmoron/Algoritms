import numpy as np
import matplotlib.pyplot as plt
from typing import List

def print_route_details(tour: List[int], distance_matrix: np.ndarray) -> None:
    """
    Печать подробной информации о маршруте.

    :param tour: Список узлов маршрута.
    :param distance_matrix: Матрица расстояний.
    """
    total_distance: float = 0.0
    print("\nПодробная информация о маршруте:")
    print("-" * 50)

    for i in range(len(tour) - 1):
        current_city: int = tour[i]
        next_city: int = tour[i + 1]
        step_distance: float = distance_matrix[current_city][next_city]
        total_distance += step_distance
        print(f"Из города {current_city} в город {next_city} расстояние: {step_distance:.2f}")

    print("-" * 50)
    print(f"Общая длина маршрута: {total_distance:.2f}")

def plot_optimization_progress(
    distances_history: List[float],
    pheromone_history: List[float],
    probabilities_history: List[float]
) -> None:
    """
    Построение графиков оптимизации.

    :param distances_history: История изменения длины маршрута.
    :param pheromone_history: История изменения уровня феромонов.
    :param probabilities_history: История изменения максимальных вероятностей.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # График длины маршрута
    ax1.plot(distances_history, 'b-', linewidth=1, label='Длина маршрута')
    ax1.set_title('Оптимизация длины маршрута')
    ax1.set_xlabel('Итерация')
    ax1.set_ylabel('Длина пути')
    ax1.grid(True)
    ax1.legend()

    # График среднего уровня феромонов
    ax2.plot(pheromone_history, 'r-', linewidth=1, label='Средний уровень феромонов')
    ax2.set_title('Динамика феромонов на лучшем маршруте')
    ax2.set_xlabel('Итерация')
    ax2.set_ylabel('Уровень феромонов')
    ax2.grid(True)
    ax2.legend()

    # График изменения максимальных вероятностей
    ax3.plot(probabilities_history, 'g-', linewidth=1, label='Макс. вероятность')
    ax3.set_title('Изменение максимальной вероятности выбора города')
    ax3.set_xlabel('Итерация')
    ax3.set_ylabel('Вероятность')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.show()

def load_distance_matrix_from_file(file_path: str, num_nodes: int) -> np.ndarray:
    """
    Загрузка матрицы расстояний из файла.

    :param file_path: Путь к файлу.
    :param num_nodes: Количество узлов.
    :return: Матрица расстояний.
    """
    # Инициализация матрицы расстояний с бесконечностью
    distance_matrix: np.ndarray = np.full((num_nodes, num_nodes), float('inf'))
    try:
        # Загрузка данных из файла
        data: np.ndarray = np.loadtxt(file_path, skiprows=1, dtype=int)

        # Заполнение матрицы расстояний
        for source, target, weight in data:
            distance_matrix[source][target] = weight
    except OSError as e:
        print(f"Ошибка загрузки файла: {e}")
        raise
    except ValueError as e:
        print(f"Ошибка формата данных в файле: {e}")
        raise

    return distance_matrix

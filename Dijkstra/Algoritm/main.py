from aco_algorithm import aco_algorithm
from utils import print_route_details, plot_optimization_progress, load_distance_matrix_from_file
import numpy as np

def main():
    print("Выберите режим работы:")
    print("1. Использовать фиксированную матрицу расстояний")
    print("2. Загрузка матрицы расстояний из файла 1000.txt")

    choice = int(input("Введите номер режима: "))

    if choice == 1:
        # Фиксированная матрица расстояний
        distance_matrix = np.array([
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            [120, 0, 22, 32, 42, 52, 602, 72, 82, 92, 102, 112],
            [140, 24, 0, 34, 44, 54, 64, 74, 84, 94, 104, 114],
            [160, 26, 36, 0, 460, 56, 66, 76, 86, 96, 106, 116],
            [180, 28, 38, 48, 0, 58, 68, 78, 88, 98, 108, 118],
            [200, 30, 40, 50, 60, 0, 70, 80, 90, 100, 110, 120],
            [22, 302, 420, 52, 62, 72, 0, 82, 92, 102, 112, 122],
            [24, 34, 44, 54, 64, 74, 84, 0, 94, 104, 114, 124],
            [26, 36, 46, 56, 66, 76, 86, 96, 0, 106, 116, 126],
            [28, 38, 48, 58, 68, 78, 88, 98, 108, 0, 118, 128],
            [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 0, 130],
            [32, 42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 0]
        ])
    elif choice == 2:
        # Загрузка матрицы из файла
        file_path = "1000.txt"
        num_nodes = 1000
        distance_matrix = load_distance_matrix_from_file(file_path, num_nodes)
    else:
        print("Неверный выбор режима.")
        return

    # Получение начальной и конечной точки
    start_point = int(input("Введите начальную точку: "))
    end_point = int(input("Введите конечную точку: "))

    # Запуск алгоритма
    iteration_distances, pheromone_history, probabilities_history, best_tour, best_distance = aco_algorithm(
        num_ants=100,
        num_iterations=100,
        evaporation_rate=0.1,
        start_point=start_point,
        end_point=end_point,
        distance_matrix=distance_matrix
    )

    # Построение графиков
    plot_optimization_progress(iteration_distances, pheromone_history, probabilities_history)

    # Вывод результатов
    print(f"\nЛучший маршрут: {best_tour}")
    print(f"Длина лучшего маршрута: {best_distance:.2f}")
    print("\nДетали маршрута:")
    print_route_details(best_tour, distance_matrix)

if __name__ == "__main__":
    main()

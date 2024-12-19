import numpy as np
import random
import matplotlib.pyplot as plt

class Ant:
    def __init__(self, num_nodes, start_point, end_point, alpha=1.0, beta=2.0):
        self.num_nodes = num_nodes
        self.start_point = start_point
        self.end_point = end_point
        self.tour = []
        self.distance = 0
        self.alpha = alpha  
        self.beta = beta    
        self.probabilities = []  

    def construct_tour(self, pheromone_matrix, visibility_matrix, distance_matrix):
        self.tour = [self.start_point]
        current = self.start_point
        available_nodes = set(range(self.num_nodes))
        available_nodes.remove(self.start_point)
        
        while current != self.end_point:
            if not available_nodes and current != self.end_point:
                self.tour.append(self.end_point)
                break
                
            cities = list(available_nodes) + [self.end_point] if self.end_point not in self.tour else list(available_nodes)
            if not cities:
                break
                
            probabilities = []
            denominator = 0
            
            #Формула
            for city in cities:
                tau = pheromone_matrix[current][city]
                eta = visibility_matrix[current][city]
                denominator += (tau ** self.alpha) * (eta ** self.beta)
            

            for next_city in cities:
                
                tau = pheromone_matrix[current][next_city]
                eta = visibility_matrix[current][next_city]
                if denominator > 0:
                    probability = ((tau ** self.alpha) * (eta ** self.beta)) / denominator
                else:
                    probability = 1.0 / len(cities)
                probabilities.append(probability)
                #ВЫвод вероятностей
                # print(f"Текущий город: {current}, Следующий город: {next_city}, Вероятность: {probability}") 
            
            next_city = np.random.choice(cities, p=probabilities)
            self.probabilities.append(max(probabilities))  
            self.tour.append(next_city)
            current = next_city
            if next_city in available_nodes:
                available_nodes.remove(next_city)
                
        self.distance = self.calculate_distance(distance_matrix)

    def calculate_distance(self, distance_matrix):
        return sum(distance_matrix[self.tour[i]][self.tour[i + 1]] 
                  for i in range(len(self.tour) - 1))

def print_route_details(tour, distance_matrix):
    total_distance = 0
    print("\nПодробная информация о маршруте:")
    print("-" * 40)
    for i in range(len(tour) - 1):
        current_city = tour[i]
        next_city = tour[i + 1]
        step_distance = distance_matrix[current_city][next_city]
        total_distance += step_distance
        print(f"Из города {current_city} в город {next_city}: расстояние = {step_distance}")
    print("-" * 40)
    print(f"Общая длина маршрута: {total_distance}")

def plot_optimization_progress(distances_history, pheromone_history, probabilities_history):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    ax1.plot(distances_history, 'b-', linewidth=1, label='Длина маршрута')
    ax1.set_title('Оптимизация длины маршрута')
    ax1.set_xlabel('Итерация')
    ax1.set_ylabel('Длина пути')
    ax1.grid(True)
    ax1.legend()
    
    pheromone_history.sort()
    ax2.plot(pheromone_history, 'r-', linewidth=1, label='Средний уровень феромонов')
    ax2.set_title('Динамика феромонов на лучшем маршруте')
    ax2.set_xlabel('Итерация')
    ax2.set_ylabel('Уровень феромонов')
    ax2.grid(True)
    ax2.legend()
    


    ax3.plot(probabilities_history, 'g-', linewidth=1, label='Макс. вероятность')
    ax3.set_title('Изменение максимальной вероятности выбора города')
    ax3.set_xlabel('Итерация')
    ax3.set_ylabel('Вероятность')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

def aco_algorithm(num_ants, num_iterations, evaporation_rate, start_point, end_point, distance_matrix):
    num_nodes = len(distance_matrix)
    pheromone_matrix = np.ones((num_nodes, num_nodes))
    visibility_matrix = np.zeros_like(distance_matrix, dtype=float)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and distance_matrix[i][j] > 0:
                visibility_matrix[i][j] = 1.0 / distance_matrix[i][j]
    
    best_distance = float('inf')
    best_tour = None
    distances_history = []
    pheromone_history = [] 
    probabilities_history = []  
    
    for iteration in range(num_iterations):
        ants = [Ant(num_nodes, start_point, end_point) for _ in range(num_ants)]
        iteration_distances = []
        iteration_probabilities = []
        
        for ant in ants:
            ant.construct_tour(pheromone_matrix, visibility_matrix, distance_matrix)
            iteration_distances.append(ant.distance)
            iteration_probabilities.extend(ant.probabilities)
            if ant.distance < best_distance:
                best_distance = ant.distance
                best_tour = ant.tour.copy()
        
        distances_history.append(np.mean(iteration_distances))
        probabilities_history.append(np.mean(iteration_probabilities))
        

        pheromone_matrix *= (1 - evaporation_rate)
        for ant in ants:
            deposit = 1.0 / ant.distance if ant.distance > 0 else 0
            for i in range(len(ant.tour) - 1):
                current = ant.tour[i]
                next_city = ant.tour[i + 1]
                pheromone_matrix[current][next_city] += deposit

        if best_tour:
            path_pheromones = [pheromone_matrix[best_tour[i]][best_tour[i+1]] 
                             for i in range(len(best_tour)-1)]
            pheromone_history.append(np.mean(path_pheromones))
    
    return distances_history, pheromone_history, probabilities_history, best_tour, best_distance


def load_distance_matrix_from_file(file_path, num_nodes):
    distance_matrix = np.full((num_nodes, num_nodes), float('inf'))  
    data = np.loadtxt(file_path, skiprows=1, dtype=int)
    

    for source, target, weight in data:
        distance_matrix[source][target] = weight  
        
    return distance_matrix

def main():
    print("Выберите режим работы:")
    print("1. Использовать фиксированную матрицу расстояний")
    print("2. Загрузить матрицу расстояний из файла")
    choice = int(input("Введите номер режима: "))
    
    if choice == 1:
        
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
        file_path = "1000.txt"
        num_nodes = 1000  
        distance_matrix = load_distance_matrix_from_file(file_path, num_nodes)
    else:
        print("Неверный выбор режима.")
        return

    start_point = int(input("Введите начальную точку: "))
    end_point = int(input("Введите конечную точку: "))

    iteration_distances, pheromone_history, probabilities_history, best_tour, best_distance = aco_algorithm(
        num_ants=100,
        num_iterations=100,
        evaporation_rate=0.1,
        start_point=start_point,
        end_point=end_point,
        distance_matrix=distance_matrix
    )

    plot_optimization_progress(iteration_distances, pheromone_history, probabilities_history)
    
    print(f"\nЛучший маршрут: {best_tour}")
    print(f"Длина лучшего маршрута: {best_distance:.2f}")
    print("\nДетали маршрута:")
    print_route_details(best_tour, distance_matrix)


if __name__ == "__main__":
    main()

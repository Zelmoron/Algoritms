package algoritm

import (
	"fmt"
	"math"
)

type GraphResult struct {
	Result map[int]int
}

// Алгоритм Дейкстры
func (gr *GraphResult) Dijkstra(graph map[int]map[int]int, start int) *GraphResult {
	// Инициализация расстояний до всех узлов
	distances := make(map[int]int)
	for node := range graph {
		distances[node] = math.MaxInt32

	}
	distances[start] = 0

	// Очередь приоритетов для посещения узлов
	queue := []int{start}

	// Проверка узлов
	visited := make(map[int]bool)

	for len(queue) > 0 {
		// Извлечение узла с минимальным расстоянием из очереди
		currentNode := queue[0]

		queue = queue[1:]
		fmt.Println(currentNode)
		// Проверка, был ли узел уже посещен
		if visited[currentNode] {
			continue
		}
		visited[currentNode] = true

		// Обновление расстояний до соседей
		for neighbor, weight := range graph[currentNode] {
			// Расчет нового расстояния

			newDistance := distances[currentNode] + weight
			fmt.Println(newDistance)
			// Если новое расстояние меньше текущего, обновляем его
			if newDistance < distances[neighbor] {
				fmt.Print(newDistance)
				fmt.Println("New")
				distances[neighbor] = newDistance
				// Добавление соседа в очередь
				queue = append(queue, neighbor)
			}
		}
	}
	gr.Result = distances
	return gr
}

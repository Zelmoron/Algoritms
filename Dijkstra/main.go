package main

import (
	algoritm "D/Algoritm"
	"fmt"
)

func main() {

	graph := map[int]map[int]int{
		1: map[int]int{2: 10, 3: 4},
		2: map[int]int{1: 10, 3: 2, 4: 5},
		3: map[int]int{1: 4, 2: 2, 4: 1},
		4: map[int]int{2: 5, 3: 1},
	}
	var g algoritm.GraphResult
	// data := graph.CreateGraf()
	// Вызов алгоритма Дейкстры
	distances := g.Dijkstra(graph, 1)

	// Вывод результатов
	fmt.Println("Расстояния до узлов из узла 1:")
	for node, distance := range distances.Result {
		fmt.Printf("Узел %d: %d\n", node, distance)
	}
}

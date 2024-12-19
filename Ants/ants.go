package main

import (
	"fmt"
	"math/rand"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

type Graph struct {
	Vertices        int
	Edges           [][]float64
	Alpha           float64
	Beta            float64
	EvaporationRate float64
	Ants            int
}

func NewGraph(vertices int, edges [][]float64, alpha, beta, evaporationRate float64, ants int) *Graph {
	return &Graph{
		Vertices:        vertices,
		Edges:           edges,
		Alpha:           alpha,
		Beta:            beta,
		EvaporationRate: evaporationRate,
		Ants:            ants,
	}
}

func (g *Graph) AntColonyOptimization(steps int) ([]int, float64, []float64) {
	// Создание графика
	p := plot.New()

	p.Title.Text = "Изменение длины пути"
	p.X.Label.Text = "Итерация"
	p.Y.Label.Text = "Длина пути"

	bestPath := []int{}
	bestLength := float64(^uint(0) >> 1) // Инициализация с большим значением
	pheromones := make([][]float64, g.Vertices)
	for i := range pheromones {
		pheromones[i] = make([]float64, g.Vertices)
	}

	lengths := make([]float64, steps)

	for step := 0; step < steps; step++ {
		allPaths := make([][]int, g.Ants)
		allLengths := make([]float64, g.Ants)

		for ant := 0; ant < g.Ants; ant++ {
			path, length := g.ConstructPath(pheromones)
			allPaths[ant] = path
			allLengths[ant] = length

			if length < bestLength {
				bestLength = length
				bestPath = path
			}
		}

		lengths[step] = bestLength // Сохраняем длину на каждом шаге

		g.UpdatePheromones(allPaths, allLengths, pheromones)
		g.EvaporatePheromones(pheromones)
	}

	// Создаем точки для графика
	pts := make(plotter.XYs, len(bestPath))
	fmt.Println(len(bestPath))
	for i := 0; i < len(bestPath); i++ {
		pts[i].X = float64(i)
		pts[i].Y = lengths[i]
	}

	line, err := plotter.NewLine(pts)
	if err != nil {
		panic(err)
	}

	p.Add(line)

	if err := p.Save(8*vg.Inch, 4*vg.Inch, "path_length.png"); err != nil {
		panic(err)
	}

	fmt.Println("График сохранен как path_length.png")
	return bestPath, bestLength, lengths
}

func (g *Graph) ConstructPath(pheromones [][]float64) ([]int, float64) {
	visited := make([]bool, g.Vertices)
	path := []int{rand.Intn(g.Vertices)}
	visited[path[0]] = true
	length := 0.0

	for len(path) < g.Vertices {
		current := path[len(path)-1]
		next := g.SelectNextVertex(current, visited, pheromones)
		length += g.Edges[current][next]
		path = append(path, next)
		visited[next] = true
	}

	length += g.Edges[path[len(path)-1]][path[0]]
	return path, length
}

func (g *Graph) SelectNextVertex(current int, visited []bool, pheromones [][]float64) int {
	probabilities := make([]float64, g.Vertices)
	sum := 0.0

	for i := 0; i < g.Vertices; i++ {
		if !visited[i] {
			probabilities[i] = pheromones[current][i] * (1.0 / g.Edges[current][i])
			sum += probabilities[i]
		}
	}

	if sum == 0 {
		return rand.Intn(g.Vertices)
	}

	for i := range probabilities {
		probabilities[i] /= sum
	}

	r := rand.Float64()
	for i, p := range probabilities {
		if r < p {
			return i
		}
		r -= p
	}

	return -1
}

func (g *Graph) UpdatePheromones(paths [][]int, lengths []float64, pheromones [][]float64) {
	for i := range pheromones {
		for j := range pheromones[i] {
			pheromones[i][j] *= (1 - g.EvaporationRate)
		}
	}

	for i, path := range paths {
		for j := 0; j < len(path)-1; j++ {
			pheromones[path[j]][path[j+1]] += 1.0 / lengths[i]
		}
		pheromones[path[len(path)-1]][path[0]] += 1.0 / lengths[i]
	}
}

func (g *Graph) EvaporatePheromones(pheromones [][]float64) {
	for i := range pheromones {
		for j := range pheromones[i] {
			pheromones[i][j] *= (1 - g.EvaporationRate)
		}
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())

	edges := [][]float64{
		{0, 10, 15, 20},
		{10, 0, 35, 25},
		{15, 35, 0, 30},
		{20, 25, 30, 0},
	}

	graph := NewGraph(4, edges, 1.0, 1.0, 0.5, 10)
	bestPath, bestLength, _ := graph.AntColonyOptimization(100)

	fmt.Printf("Лучший путь: %v, Длина: %v\n", bestPath, bestLength)

}

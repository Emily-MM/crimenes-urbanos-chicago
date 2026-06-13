package main

import (
	"math"
	"math/rand"
	"sync"
)

type Sample struct {
	Features []float64
	Target   float64
	Weight   float64
}

type LinearModel struct {
	Weights []float64
	mu      sync.Mutex
}

func NewModel(n int) *LinearModel {
	w := make([]float64, n+1)
	for i := range w {
		w[i] = rand.Float64()*0.01 - 0.005
	}
	return &LinearModel{Weights: w}
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (m *LinearModel) Predict(f []float64) float64 {
	z := m.Weights[0]
	for i, v := range f {
		z += m.Weights[i+1] * v
	}
	return sigmoid(z)
}

package main

import (
	"math"
	"sync"
)

type Sample struct {
	Features []float64 `json:"features"`
	Target   float64   `json:"target"`
	Weight   float64   `json:"weight"`
}

type InitMessage struct {
	Type    string   `json:"type"`
	Samples []Sample `json:"samples"`
}

type InitStartMessage struct {
	Type  string `json:"type"`
	Total int    `json:"total"`
}

type InitChunkMessage struct {
	Type    string   `json:"type"`
	Samples []Sample `json:"samples"`
}

type InitEndMessage struct {
	Type string `json:"type"`
}

type EvaluateMessage struct {
	Type    string    `json:"type"`
	Weights []float64 `json:"weights"`
}

type TrainMessage struct {
	Type    string    `json:"type"`
	Weights []float64 `json:"weights"`
}

type GradientResponse struct {
	Gradients []float64 `json:"gradients"`
	Loss      float64   `json:"loss"`
	Count     int       `json:"count"`
	NodeID    string    `json:"node_id"`
}

type EvalResponse struct {
	TP     int    `json:"tp"`
	TN     int    `json:"tn"`
	FP     int    `json:"fp"`
	FN     int    `json:"fn"`
	NodeID string `json:"node_id"`
}

var (
	nodeID      string
	myPartition []Sample
)

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func predict(weights []float64, features []float64) float64 {
	z := weights[0]
	for i, v := range features {
		z += weights[i+1] * v
	}
	return sigmoid(z)
}

const numGoroutines = 8

type gradResult struct {
	Gradients []float64
	Loss      float64
	Count     int
}

func gradientSubWorker(weights []float64, subPart []Sample, resultCh chan<- gradResult, wg *sync.WaitGroup) {
	defer wg.Done()

	numW := len(weights)
	grads := make([]float64, numW)
	loss := 0.0

	for _, s := range subPart {
		pred := predict(weights, s.Features)
		if pred < 1e-10 {
			pred = 1e-10
		}
		if pred > 1-1e-10 {
			pred = 1 - 1e-10
		}

		loss += s.Weight * -(s.Target*math.Log(pred) + (1-s.Target)*math.Log(1-pred))

		err := s.Weight * (pred - s.Target)
		grads[0] += err
		for i, v := range s.Features {
			grads[i+1] += err * v
		}
	}

	resultCh <- gradResult{Gradients: grads, Loss: loss, Count: len(subPart)}
}

func calcularGradiente(weights []float64, partition []Sample) ([]float64, float64) {
	numW := len(weights)
	n := len(partition)

	workers := numGoroutines
	if n < workers {
		workers = n
	}
	if workers == 0 {
		return make([]float64, numW), 0
	}

	subSize := n / workers
	resultCh := make(chan gradResult, workers)
	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {
		start := i * subSize
		end := start + subSize
		if i == workers-1 {
			end = n
		}
		wg.Add(1)
		go gradientSubWorker(weights, partition[start:end], resultCh, &wg)
	}

	go func() {
		wg.Wait()
		close(resultCh)
	}()

	aggGrads := make([]float64, numW)
	totalLoss := 0.0
	totalN := 0

	for r := range resultCh {
		for i, g := range r.Gradients {
			aggGrads[i] += g
		}
		totalLoss += r.Loss
		totalN += r.Count
	}

	for i := range aggGrads {
		aggGrads[i] /= float64(totalN)
	}

	return aggGrads, totalLoss / float64(totalN)
}

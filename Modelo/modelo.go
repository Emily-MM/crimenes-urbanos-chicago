package main

import (
	"fmt"
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

type riskKey struct{ Hour, District int }

func calcularUmbrales(records []rawRecord) map[riskKey]bool {
	counts := make(map[riskKey]int)
	for _, r := range records {
		counts[riskKey{r.Hour, r.District}]++
	}

	valores := make([]int, 0, len(counts))
	for _, v := range counts {
		valores = append(valores, v)
	}

	for i := 0; i < len(valores); i++ {
		for j := i + 1; j < len(valores); j++ {
			if valores[j] < valores[i] {
				valores[i], valores[j] = valores[j], valores[i]
			}
		}
	}

	p75idx := int(float64(len(valores)) * 0.75)
	umbral := float64(valores[p75idx])

	altoRiesgo := make(map[riskKey]bool)
	for k, v := range counts {
		if float64(v) >= umbral {
			altoRiesgo[k] = true
		}
	}

	altoCount := len(altoRiesgo)
	fmt.Printf("  umbral percentil 75                    : %.0f crimenes\n", umbral)
	fmt.Printf("  combinaciones de alto riesgo           : %d / %d (%.0f%%)\n",
		altoCount, len(counts), float64(altoCount)/float64(len(counts))*100)
	return altoRiesgo
}

var tiposCrimen = map[string]float64{
	"THEFT": 0, "BATTERY": 1, "CRIMINAL DAMAGE": 2, "NARCOTICS": 3,
	"ASSAULT": 4, "BURGLARY": 5, "MOTOR VEHICLE THEFT": 6, "ROBBERY": 7,
	"DECEPTIVE PRACTICE": 8, "OTHER OFFENSE": 9,
}

func encodeTipo(t string) float64 {
	if v, ok := tiposCrimen[t]; ok {
		return v / 9.0
	}
	return 0.5
}

type rawRecord struct {
	Hour          int
	District      int
	PrimaryType   string
	Domestic      bool
	CommunityArea int
	Year          int
	Latitude      float64
	Longitude     float64
}

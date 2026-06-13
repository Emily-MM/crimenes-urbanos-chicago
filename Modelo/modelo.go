package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
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

func loadRecords(path string, numWorkers int) ([]rawRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	type chunk [][]string
	jobs := make(chan []string, numWorkers*200)
	results := make(chan []rawRecord, numWorkers)
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var batch []rawRecord
			for row := range jobs {
				if len(row) < 18 {
					continue
				}
				hour, _ := strconv.Atoi(row[3])
				district, _ := strconv.Atoi(row[11])
				area, _ := strconv.Atoi(row[13])
				lat, err1 := strconv.ParseFloat(row[16], 64)
				lng, err2 := strconv.ParseFloat(row[17], 64)
				if err1 != nil || err2 != nil || lat == 0 || lng == 0 {
					continue
				}
				year, _ := strconv.Atoi(row[15])
				batch = append(batch, rawRecord{
					Hour:          hour,
					District:      district,
					PrimaryType:   row[5],
					Domestic:      row[9] == "true",
					CommunityArea: area,
					Year:          year,
					Latitude:      lat,
					Longitude:     lng,
				})
			}
			results <- batch
		}()
	}

	go func() {
		reader := csv.NewReader(bufio.NewReaderSize(file, 4*1024*1024))
		reader.Read()
		for {
			row, err := reader.Read()
			if err == io.EOF {
				break
			}
			if err == nil {
				jobs <- row
			}
		}
		close(jobs)
	}()

	go func() {
		wg.Wait()
		close(results)
	}()

	var all []rawRecord
	for batch := range results {
		all = append(all, batch...)
	}
	return all, nil
}

func buildDataset(records []rawRecord, altoRiesgo map[riskKey]bool) []Sample {
	dataset := make([]Sample, len(records))
	for i, r := range records {
		h := float64(r.Hour) / 23.0
		d := float64(r.District) / 25.0
		area := float64(r.CommunityArea) / 77.0
		domestic := 0.0
		if r.Domestic {
			domestic = 1.0
		}
		esNoche := 0.0
		if r.Hour >= 20 || r.Hour <= 5 {
			esNoche = 1.0
		}
		esTarde := 0.0
		if r.Hour >= 12 && r.Hour < 20 {
			esTarde = 1.0
		}

		target := 0.0
		if altoRiesgo[riskKey{r.Hour, r.District}] {
			target = 1.0
		}

		yr := (float64(r.Year) - 2001.0) / 25.0

		weight := 1.0
		if target == 1.0 {
			weight = 1.35
		}

		dataset[i] = Sample{
			Features: []float64{
				h, d, area,
				encodeTipo(r.PrimaryType),
				domestic, esNoche, esTarde,
				h * d, yr,
			},
			Target: target,
			Weight: weight,
		}
	}
	return dataset
}

type PartialGradient struct {
	Gradients []float64
	Loss      float64
	Count     int
}

func gradientWorker(
	partition []Sample,
	model *LinearModel,
	resultCh chan<- PartialGradient,
	wg *sync.WaitGroup,
) {
	defer wg.Done()

	nw := len(model.Weights)
	grads := make([]float64, nw)
	loss := 0.0

	for _, s := range partition {
		pred := model.Predict(s.Features)
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

	n := float64(len(partition))
	for i := range grads {
		grads[i] /= n
	}

	resultCh <- PartialGradient{
		Gradients: grads,
		Loss:      loss / n,
		Count:     len(partition),
	}
}

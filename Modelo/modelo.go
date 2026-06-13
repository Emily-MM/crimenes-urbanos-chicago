package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"
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

func train(dataset []Sample, model *LinearModel, numWorkers, epochs int, lr float64) {
	fmt.Printf("\nentrenamiento paralelo\n")
	fmt.Printf("  workers : %d\n", numWorkers)
	fmt.Printf("  epocas  : %d\n", epochs)
	fmt.Printf("  muestras: %d\n", len(dataset))
	fmt.Printf("  lr      : %.4f\n\n", lr)

	partSize := len(dataset) / numWorkers

	for epoch := 0; epoch < epochs; epoch++ {
		resultCh := make(chan PartialGradient, numWorkers)
		var wg sync.WaitGroup

		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			start := i * partSize
			end := start + partSize
			if i == numWorkers-1 {
				end = len(dataset)
			}
			go gradientWorker(dataset[start:end], model, resultCh, &wg)
		}

		go func() {
			wg.Wait()
			close(resultCh)
		}()

		nw := len(model.Weights)
		aggGrads := make([]float64, nw)
		totalLoss := 0.0
		totalN := 0

		for pg := range resultCh {
			for i, g := range pg.Gradients {
				aggGrads[i] += g * float64(pg.Count)
			}
			totalLoss += pg.Loss * float64(pg.Count)
			totalN += pg.Count
		}

		for i := range aggGrads {
			aggGrads[i] /= float64(totalN)
		}

		model.mu.Lock()
		for i := range model.Weights {
			model.Weights[i] -= lr * aggGrads[i]
		}
		model.mu.Unlock()

		if epoch%10 == 0 || epoch == epochs-1 {
			avgLoss := totalLoss / float64(totalN)
			fmt.Printf("  [%d/%d] loss %.4f\n", epoch+1, epochs, avgLoss)
		}
	}
}

func evaluate(model *LinearModel, dataset []Sample) {
	tp, tn, fp, fn := 0, 0, 0, 0

	for _, s := range dataset {
		pred := model.Predict(s.Features)
		predLabel := 0.0
		if pred >= 0.5 {
			predLabel = 1.0
		}

		switch {
		case predLabel == 1 && s.Target == 1:
			tp++
		case predLabel == 0 && s.Target == 0:
			tn++
		case predLabel == 1 && s.Target == 0:
			fp++
		case predLabel == 0 && s.Target == 1:
			fn++
		}
	}

	total := tp + tn + fp + fn
	precision := 0.0
	recall := 0.0
	f1 := 0.0

	if tp+fp > 0 {
		precision = float64(tp) / float64(tp+fp) * 100
	}
	if tp+fn > 0 {
		recall = float64(tp) / float64(tp+fn) * 100
	}
	if precision+recall > 0 {
		f1 = 2 * (precision * recall) / (precision + recall)
	}

	accuracy := float64(tp+tn) / float64(total) * 100

	fmt.Println("\nmetricas de clasificacion")
	fmt.Println("────────────────────────────────────────")
	fmt.Printf("accuracy  : %.2f%%  (%d/%d correctos)\n", accuracy, tp+tn, total)
	fmt.Printf("precision : %.2f%%  \n", precision)
	fmt.Printf("recall    : %.2f%%  \n", recall)
	fmt.Printf("F1        : %.2f\n", f1)
	fmt.Println()
	fmt.Printf("  verdaderos positivos : %d\n", tp)
	fmt.Printf("  verdaderos negativos : %d\n", tn)
	fmt.Printf("  falsos positivos : %d\n", fp)
	fmt.Printf("  falsos negativos : %d\n", fn)
}

func benchmark(dataset []Sample, counts []int, epochs int, lr float64) int {
	fmt.Println("\nbenchmark workers")
	fmt.Println("────────────────────────────────────────────────")
	fmt.Printf("%-10s %-15s %-10s\n", "workers", "tiempo", "speedup")
	fmt.Println("────────────────────────────────────────────────")

	var baseTime time.Duration
	bestWorkers := counts[0]
	bestTime := time.Duration(math.MaxInt64)

	for _, nw := range counts {
		m := NewModel(9)
		start := time.Now()
		trainSilent(dataset, m, nw, epochs, lr)
		elapsed := time.Since(start)

		if baseTime == 0 {
			baseTime = elapsed
		}
		speedup := float64(baseTime) / float64(elapsed)
		fmt.Printf("%-10d %-15s %.2fx\n", nw, elapsed.Round(time.Millisecond), speedup)

		if elapsed < bestTime {
			bestTime = elapsed
			bestWorkers = nw
		}
	}

	fmt.Printf("\nworkers optimo: %d (%s)\n", bestWorkers, bestTime.Round(time.Millisecond))
	return bestWorkers
}

func trainSilent(dataset []Sample, model *LinearModel, numWorkers, epochs int, lr float64) {
	partSize := len(dataset) / numWorkers
	if partSize == 0 {
		partSize = 1
		numWorkers = len(dataset)
	}
	for epoch := 0; epoch < epochs; epoch++ {
		resultCh := make(chan PartialGradient, numWorkers)
		var wg sync.WaitGroup
		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			start := i * partSize
			end := start + partSize
			if i == numWorkers-1 {
				end = len(dataset)
			}
			if start >= len(dataset) {
				wg.Done()
				continue
			}
			go gradientWorker(dataset[start:end], model, resultCh, &wg)
		}
		go func() { wg.Wait(); close(resultCh) }()
		nw := len(model.Weights)
		aggGrads := make([]float64, nw)
		totalN := 0
		for pg := range resultCh {
			for i, g := range pg.Gradients {
				aggGrads[i] += g * float64(pg.Count)
			}
			totalN += pg.Count
		}
		for i := range aggGrads {
			aggGrads[i] /= float64(totalN)
		}
		model.mu.Lock()
		for i := range model.Weights {
			model.Weights[i] -= lr * aggGrads[i]
		}
		model.mu.Unlock()
	}
}

type ModelJSON struct {
	Weights     []float64 `json:"weights"`
	NumFeatures int       `json:"num_features"`
	TrainedAt   string    `json:"trained_at"`
}

func saveModel(model *LinearModel, path string) error {
	mj := ModelJSON{
		Weights:     model.Weights,
		NumFeatures: len(model.Weights) - 1,
		TrainedAt:   time.Now().Format(time.RFC3339),
	}
	data, err := json.MarshalIndent(mj, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func main() {
	inputPath := "../datos/crimenes_limpio.csv"
	modelPath := "model.json"
	numWorkers := 8
	epochs := 100
	lr := 0.3

	fmt.Printf("\ncargando %s\n", inputPath)
	start := time.Now()
	records, err := loadRecords(inputPath, numWorkers)
	if err != nil {
		fmt.Println("error:", err)
		os.Exit(1)
	}
	fmt.Printf("  %d registros en %s\n", len(records), time.Since(start).Round(time.Millisecond))

	fmt.Println("\ncalculando umbrales de riesgo...")
	altoRiesgo := calcularUmbrales(records)

	fmt.Println("\nconstruyendo dataset de entrenamiento...")
	startDS := time.Now()
	dataset := buildDataset(records, altoRiesgo)
	positivos := 0
	for _, s := range dataset {
		if s.Target == 1 {
			positivos++
		}
	}
	fmt.Printf("  %d muestras construidas en %s\n", len(dataset), time.Since(startDS).Round(time.Millisecond))
	fmt.Printf("  alto riesgo : %d (%.1f%%)\n", positivos, float64(positivos)/float64(len(dataset))*100)
	fmt.Printf("  bajo riesgo : %d (%.1f%%)\n", len(dataset)-positivos, float64(len(dataset)-positivos)/float64(len(dataset))*100)

	subSize := 100000
	if len(dataset) < subSize {
		subSize = len(dataset)
	}
	sub := dataset[:subSize]
	fmt.Printf("\nbenchmark sobre %d muestras\n", subSize)
	optWorkers := benchmark(sub, []int{1, 2, 4, 8, 16, 32}, 5, lr)

	fmt.Println("\n---")
	fmt.Printf("entrenamiento final (%d workers)\n", optWorkers)
	fmt.Println("---")
	model := NewModel(9)
	startTrain := time.Now()
	train(dataset, model, optWorkers, epochs, lr)
	fmt.Printf("completado en %s\n", time.Since(startTrain).Round(time.Millisecond))

	evaluate(model, dataset)

	if err := saveModel(model, modelPath); err != nil {
		fmt.Println("error guardando modelo:", err)
	} else {
		fmt.Printf("modelo guardado en %s\n", modelPath)
	}
}

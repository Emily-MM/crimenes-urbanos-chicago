package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Crime struct {
	ID                  int
	CaseNumber          string
	Date                time.Time
	Hour                int
	Block               string
	PrimaryType         string
	Description         string
	LocationDescription string
	Arrest              bool
	Domestic            bool
	Beat                int
	District            int
	Ward                int
	CommunityArea       int
	FBICode             string
	Year                int
	Latitude            float64
	Longitude           float64
}

type WorkerResult struct {
	Records  []Crime
	Errors   int
	WorkerID int
}

func parseRow(row []string) (Crime, error) {
	if len(row) < 22 {
		return Crime{}, fmt.Errorf("fila incompleta: %d columnas", len(row))
	}

	id, err := strconv.Atoi(strings.TrimSpace(row[0]))
	if err != nil {
		return Crime{}, fmt.Errorf("ID invalido: %s", row[0])
	}

	date, err := time.Parse("01/02/2006 03:04:05 PM", strings.TrimSpace(row[2]))
	if err != nil {
		return Crime{}, fmt.Errorf("fecha invalida: %s", row[2])
	}

	lat, err := strconv.ParseFloat(strings.TrimSpace(row[19]), 64)
	if err != nil || lat == 0 {
		return Crime{}, fmt.Errorf("latitud invalida")
	}
	lng, err := strconv.ParseFloat(strings.TrimSpace(row[20]), 64)
	if err != nil || lng == 0 {
		return Crime{}, fmt.Errorf("longitud invalida")
	}
	if lat < 41.6 || lat > 42.1 || lng < -87.9 || lng > -87.5 {
		return Crime{}, fmt.Errorf("coordenadas fuera de Chicago: %.4f, %.4f", lat, lng)
	}

	beat, _ := strconv.Atoi(strings.TrimSpace(row[10]))
	district, _ := strconv.Atoi(strings.TrimSpace(row[11]))
	ward, _ := strconv.Atoi(strings.TrimSpace(row[12]))
	communityArea, _ := strconv.Atoi(strings.TrimSpace(row[13]))
	year, _ := strconv.Atoi(strings.TrimSpace(row[17]))

	arrest := strings.ToLower(strings.TrimSpace(row[8])) == "true"
	domestic := strings.ToLower(strings.TrimSpace(row[9])) == "true"

	primaryType := strings.TrimSpace(row[5])
	if primaryType == "" {
		return Crime{}, fmt.Errorf("Primary Type vacio")
	}

	return Crime{
		ID:                  id,
		CaseNumber:          strings.TrimSpace(row[1]),
		Date:                date,
		Hour:                date.Hour(),
		Block:               strings.TrimSpace(row[3]),
		PrimaryType:         primaryType,
		Description:         strings.TrimSpace(row[6]),
		LocationDescription: strings.TrimSpace(row[7]),
		Arrest:              arrest,
		Domestic:            domestic,
		Beat:                beat,
		District:            district,
		Ward:                ward,
		CommunityArea:       communityArea,
		FBICode:             strings.TrimSpace(row[14]),
		Year:                year,
		Latitude:            lat,
		Longitude:           lng,
	}, nil
}

func worker(id int, jobs <-chan []string, results chan<- WorkerResult, wg *sync.WaitGroup) {
	defer wg.Done()

	result := WorkerResult{WorkerID: id}

	for row := range jobs {
		crime, err := parseRow(row)
		if err != nil {
			result.Errors++
			continue
		}
		result.Records = append(result.Records, crime)
	}

	results <- result
}

func loadConcurrent(filePath string, numWorkers int) ([]Crime, int, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, 0, fmt.Errorf("no se pudo abrir el archivo: %w", err)
	}
	defer file.Close()

	jobs := make(chan []string, numWorkers*100)
	results := make(chan WorkerResult, numWorkers)

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go worker(i, jobs, results, &wg)
	}

	go func() {
		reader := csv.NewReader(bufio.NewReaderSize(file, 1024*1024))
		reader.Read()

		for {
			row, err := reader.Read()
			if err != nil {
				break
			}
			jobs <- row
		}
		close(jobs)
	}()

	go func() {
		wg.Wait()
		close(results)
	}()

	var allCrimes []Crime
	totalErrors := 0

	for result := range results {
		allCrimes = append(allCrimes, result.Records...)
		totalErrors += result.Errors
		fmt.Printf("  [Worker %d] proceso %d registros, %d errores\n",
			result.WorkerID, len(result.Records), result.Errors)
	}

	return allCrimes, totalErrors, nil
}

func analyze(crimes []Crime) {
	if len(crimes) == 0 {
		fmt.Println("Sin datos para analizar.")
		return
	}

	typeCounts := make(map[string]int)
	arrestCount := 0
	districtCounts := make(map[int]int)
	hourCounts := make(map[int]int)

	for _, c := range crimes {
		typeCounts[c.PrimaryType]++
		districtCounts[c.District]++
		hourCounts[c.Hour]++
		if c.Arrest {
			arrestCount++
		}
	}

	total := len(crimes)
	fmt.Printf("\nANALISIS DE DATOS LIMPIOS\n")
	fmt.Printf("════════════════════════════════════════\n")
	fmt.Printf("Total registros validos : %d\n", total)
	fmt.Printf("Tasa de arresto         : %.2f%%\n", float64(arrestCount)/float64(total)*100)

	fmt.Printf("\nTop 5 tipos de crimen:\n")
	printTopN(typeCounts, 5)

	fmt.Printf("\nTop 5 distritos con mas crimenes:\n")
	printTopNInt(districtCounts, 5)

	fmt.Printf("\nHoras pico (top 3):\n")
	printTopNInt(hourCounts, 3)
}

func printTopN(m map[string]int, n int) {
	for i := 0; i < n; i++ {
		maxKey, maxVal := "", 0
		for k, v := range m {
			if v > maxVal {
				maxKey, maxVal = k, v
			}
		}
		if maxKey == "" {
			break
		}
		fmt.Printf("  %-35s %d\n", maxKey, maxVal)
		delete(m, maxKey)
	}
}

func printTopNInt(m map[int]int, n int) {
	for i := 0; i < n; i++ {
		maxKey, maxVal := math.MinInt32, 0
		for k, v := range m {
			if v > maxVal {
				maxKey, maxVal = k, v
			}
		}
		if maxVal == 0 {
			break
		}
		fmt.Printf("  Distrito/Hora %-5d → %d crimenes\n", maxKey, maxVal)
		delete(m, maxKey)
	}
}

func saveCSV(outputPath string, crimes []Crime, numWorkers int) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("no se pudo crear %s: %w", outputPath, err)
	}
	defer file.Close()

	writer := csv.NewWriter(bufio.NewWriterSize(file, 4*1024*1024))
	defer writer.Flush()

	writer.Write([]string{
		"ID", "CaseNumber", "Date", "Hour", "Block",
		"PrimaryType", "Description", "LocationDescription",
		"Arrest", "Domestic", "Beat", "District", "Ward",
		"CommunityArea", "FBICode", "Year", "Latitude", "Longitude",
	})

	blockSize := len(crimes) / numWorkers
	type block struct {
		rows [][]string
	}

	resultsCh := make(chan block, numWorkers)
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		start := i * blockSize
		end := start + blockSize
		if i == numWorkers-1 {
			end = len(crimes)
		}

		go func(slice []Crime) {
			defer wg.Done()
			rows := make([][]string, 0, len(slice))
			for _, c := range slice {
				rows = append(rows, []string{
					strconv.Itoa(c.ID),
					c.CaseNumber,
					c.Date.Format("2006-01-02 15:04:05"),
					strconv.Itoa(c.Hour),
					c.Block,
					c.PrimaryType,
					c.Description,
					c.LocationDescription,
					strconv.FormatBool(c.Arrest),
					strconv.FormatBool(c.Domestic),
					strconv.Itoa(c.Beat),
					strconv.Itoa(c.District),
					strconv.Itoa(c.Ward),
					strconv.Itoa(c.CommunityArea),
					c.FBICode,
					strconv.Itoa(c.Year),
					strconv.FormatFloat(c.Latitude, 'f', 8, 64),
					strconv.FormatFloat(c.Longitude, 'f', 8, 64),
				})
			}
			resultsCh <- block{rows}
		}(crimes[start:end])
	}

	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	written := 0
	for b := range resultsCh {
		for _, row := range b.rows {
			writer.Write(row)
		}
		written += len(b.rows)
		fmt.Printf("\r  Guardando... %d / %d filas", written, len(crimes))
	}
	fmt.Println()

	return nil
}

func main() {
	filePath := "../datos/Crimes_-_2001_to_Present_20260603.csv"
	outputPath := "crimenes_limpio.csv"
	numWorkers := 8

	fmt.Printf("Iniciando carga concurrente con %d workers...\n", numWorkers)
	fmt.Printf("   Archivo: %s\n\n", filePath)

	start := time.Now()

	crimes, errors, err := loadConcurrent(filePath, numWorkers)
	if err != nil {
		fmt.Println("Error:", err)
		os.Exit(1)
	}

	elapsed := time.Since(start)
	fmt.Printf("\nCarga completada en %s\n", elapsed)
	fmt.Printf("   Registros validos    : %d\n", len(crimes))
	fmt.Printf("   Registros descartados: %d\n", errors)

	analyze(crimes)

	fmt.Printf("\nGuardando CSV limpio en: %s\n", outputPath)
	startSave := time.Now()
	if err := saveCSV(outputPath, crimes, numWorkers); err != nil {
		fmt.Println("Error guardando:", err)
		os.Exit(1)
	}
	fmt.Printf("CSV guardado en %s\n", time.Since(startSave))
	fmt.Printf("   Columnas: ID, CaseNumber, Date, Hour, Block, PrimaryType,\n")
	fmt.Printf("             Description, LocationDescription, Arrest, Domestic,\n")
	fmt.Printf("             Beat, District, Ward, CommunityArea, FBICode, Year,\n")
	fmt.Printf("             Latitude, Longitude\n")
}

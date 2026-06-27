// Package main implementa la API coordinadora del sistema distribuido
// de prediccion de riesgo de delitos en Chicago.
//
//	@title			Chicago Crime Risk API
//	@version		1.0
//	@description	API coordinadora del cluster de nodos ML para prediccion
//	@description	de riesgo de delitos por hora y distrito en Chicago.
//	@description	Distribuye el entrenamiento entre 3 nodos via TCP (topologia Star)
//	@description	y expone predicciones en tiempo real. Persiste historial de
//	@description	entrenamientos y predicciones en MongoDB.
//	@host			localhost:8080
//	@BasePath		/
//	@securityDefinitions.apikey	BearerAuth
//	@in							header
//	@name						Authorization
//	@description				Token JWT obtenido en /login. Formato: "Bearer {token}"
package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"os"
	"strconv"
	"sync"
	"time"
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

type EvaluateMessage struct {
	Type    string    `json:"type"`
	Weights []float64 `json:"weights"`
}

type EvalResponse struct {
	TP     int    `json:"tp"`
	TN     int    `json:"tn"`
	FP     int    `json:"fp"`
	FN     int    `json:"fn"`
	NodeID string `json:"node_id"`
}

type NodeConn struct {
	Address string
	Conn    net.Conn
	Reader  *bufio.Reader
	Encoder *json.Encoder
	Alive   bool
	mu      sync.Mutex
}

func connectToNode(address string) (*NodeConn, error) {
	conn, err := net.DialTimeout("tcp", address, 5*time.Second)
	if err != nil {
		return nil, err
	}
	return &NodeConn{
		Address: address,
		Conn:    conn,
		Reader:  bufio.NewReader(conn),
		Encoder: json.NewEncoder(conn),
		Alive:   true,
	}, nil
}

func (nc *NodeConn) sendInit(samples []Sample) error {
	nc.mu.Lock()
	defer nc.mu.Unlock()

	const chunkSize = 50000
	total := len(samples)

	startMsg := InitStartMessage{Type: "init_start", Total: total}
	if err := nc.Encoder.Encode(startMsg); err != nil {
		return err
	}
	if _, err := nc.Reader.ReadBytes('\n'); err != nil {
		return err
	}

	for start := 0; start < total; start += chunkSize {
		end := start + chunkSize
		if end > total {
			end = total
		}
		chunk := InitChunkMessage{Type: "init_chunk", Samples: samples[start:end]}
		if err := nc.Encoder.Encode(chunk); err != nil {
			return err
		}
	}

	endMsg := InitEndMessage{Type: "init_end"}
	if err := nc.Encoder.Encode(endMsg); err != nil {
		return err
	}

	line, err := nc.Reader.ReadBytes('\n')
	if err != nil {
		return err
	}
	var ack map[string]string
	return json.Unmarshal(line, &ack)
}

func (nc *NodeConn) reconnect() error {
	nc.mu.Lock()
	defer nc.mu.Unlock()

	if nc.Conn != nil {
		nc.Conn.Close()
	}

	conn, err := net.DialTimeout("tcp", nc.Address, 5*time.Second)
	if err != nil {
		return err
	}

	nc.Conn = conn
	nc.Reader = bufio.NewReader(conn)
	nc.Encoder = json.NewEncoder(conn)
	nc.Alive = true
	return nil
}

func (nc *NodeConn) requestGradient(weights []float64, timeout time.Duration) (*GradientResponse, error) {
	nc.mu.Lock()
	defer nc.mu.Unlock()

	msg := TrainMessage{Type: "train", Weights: weights}
	if err := nc.Encoder.Encode(msg); err != nil {
		nc.Alive = false
		return nil, err
	}

	nc.Conn.SetReadDeadline(time.Now().Add(timeout))
	line, err := nc.Reader.ReadBytes('\n')
	nc.Conn.SetReadDeadline(time.Time{})

	if err != nil {
		nc.Alive = false
		return nil, fmt.Errorf("nodo %s no respondio a tiempo: %w", nc.Address, err)
	}

	var resp GradientResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (nc *NodeConn) requestEval(weights []float64, timeout time.Duration) (*EvalResponse, error) {
	nc.mu.Lock()
	defer nc.mu.Unlock()

	msg := EvaluateMessage{Type: "evaluate", Weights: weights}
	if err := nc.Encoder.Encode(msg); err != nil {
		nc.Alive = false
		return nil, err
	}

	nc.Conn.SetReadDeadline(time.Now().Add(timeout))
	line, err := nc.Reader.ReadBytes('\n')
	nc.Conn.SetReadDeadline(time.Time{})

	if err != nil {
		nc.Alive = false
		return nil, fmt.Errorf("nodo %s no respondio a tiempo: %w", nc.Address, err)
	}

	var resp EvalResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

type LinearModel struct {
	Weights []float64
}

func NewModel(n int) *LinearModel {
	w := make([]float64, n+1)
	for i := range w {
		w[i] = 0.005
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

type rawRecord struct {
	Hour, District, CommunityArea, Year int
	PrimaryType                         string
	Domestic                            bool
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

func loadRecords(path string) ([]rawRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(bufio.NewReaderSize(file, 4*1024*1024))
	reader.Read()

	var records []rawRecord
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil || len(row) < 18 {
			continue
		}
		hour, _ := strconv.Atoi(row[3])
		district, _ := strconv.Atoi(row[11])
		area, _ := strconv.Atoi(row[13])
		year, _ := strconv.Atoi(row[15])
		records = append(records, rawRecord{
			Hour: hour, District: district, CommunityArea: area,
			Year: year, PrimaryType: row[5], Domestic: row[9] == "true",
		})
	}
	return records, nil
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
	umbral := float64(valores[int(float64(len(valores))*0.75)])
	altoRiesgo := make(map[riskKey]bool)
	for k, v := range counts {
		if float64(v) >= umbral {
			altoRiesgo[k] = true
		}
	}
	return altoRiesgo
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
		yr := (float64(r.Year) - 2001.0) / 25.0

		target := 0.0
		if altoRiesgo[riskKey{r.Hour, r.District}] {
			target = 1.0
		}
		weight := 1.0
		if target == 1.0 {
			weight = 1.35
		}

		dataset[i] = Sample{
			Features: []float64{h, d, area, encodeTipo(r.PrimaryType), domestic, esNoche, esTarde, h * d, yr},
			Target:   target,
			Weight:   weight,
		}
	}
	return dataset
}

type Coordinator struct {
	Nodes []*NodeConn
	Model *LinearModel
	mu    sync.Mutex
}

func NewCoordinator(addresses []string) (*Coordinator, error) {
	var nodes []*NodeConn
	for _, addr := range addresses {
		nc, err := connectToNode(addr)
		if err != nil {
			log.Printf("aviso: no se pudo conectar a %s: %v\n", addr, err)
			continue
		}
		nodes = append(nodes, nc)
	}
	if len(nodes) == 0 {
		return nil, fmt.Errorf("no se pudo conectar a ningun nodo")
	}
	return &Coordinator{Nodes: nodes, Model: NewModel(9)}, nil
}

func (c *Coordinator) initNodes(dataset []Sample) error {
	numNodes := len(c.Nodes)
	partSize := len(dataset) / numNodes

	var wg sync.WaitGroup
	errCh := make(chan error, numNodes)

	for i, node := range c.Nodes {
		start := i * partSize
		end := start + partSize
		if i == numNodes-1 {
			end = len(dataset)
		}
		partition := dataset[start:end]

		wg.Add(1)
		go func(n *NodeConn, part []Sample) {
			defer wg.Done()
			if err := n.sendInit(part); err != nil {
				errCh <- fmt.Errorf("nodo %s: %w", n.Address, err)
				return
			}
			fmt.Printf("nodo %s inicializado con %d muestras\n", n.Address, len(part))
		}(node, partition)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		log.Println("error en init:", err)
	}
	return nil
}

func (c *Coordinator) trainEpoch(lr float64, timeout time.Duration) (float64, int) {
	type result struct {
		resp *GradientResponse
		err  error
	}

	resultCh := make(chan result, len(c.Nodes))
	var wg sync.WaitGroup

	currentWeights := make([]float64, len(c.Model.Weights))
	copy(currentWeights, c.Model.Weights)

	for _, node := range c.Nodes {
		if !node.Alive {
			if err := node.reconnect(); err != nil {
				log.Printf("nodo %s sigue caido: %v\n", node.Address, err)
			} else {
				log.Printf("nodo %s reconectado\n", node.Address)
			}
		}
	}

	for _, node := range c.Nodes {
		if !node.Alive {
			continue
		}
		wg.Add(1)
		go func(n *NodeConn) {
			defer wg.Done()
			resp, err := n.requestGradient(currentWeights, timeout)
			resultCh <- result{resp: resp, err: err}
		}(node)
	}

	go func() {
		wg.Wait()
		close(resultCh)
	}()

	numW := len(c.Model.Weights)
	aggGrads := make([]float64, numW)
	totalLoss := 0.0
	totalN := 0
	nodosActivos := 0

	for r := range resultCh {
		if r.err != nil {
			log.Println("nodo no respondio:", r.err)
			continue
		}
		nodosActivos++
		for i, g := range r.resp.Gradients {
			aggGrads[i] += g * float64(r.resp.Count)
		}
		totalLoss += r.resp.Loss * float64(r.resp.Count)
		totalN += r.resp.Count
	}

	if totalN == 0 {
		return 0, 0
	}

	for i := range aggGrads {
		aggGrads[i] /= float64(totalN)
	}

	c.mu.Lock()
	for i := range c.Model.Weights {
		c.Model.Weights[i] -= lr * aggGrads[i]
	}
	c.mu.Unlock()

	return totalLoss / float64(totalN), nodosActivos
}

type EvalMetrics struct {
	Accuracy, Precision, Recall, F1 float64
	TP, TN, FP, FN                  int
}

func (c *Coordinator) evaluate(timeout time.Duration) EvalMetrics {
	type result struct {
		resp *EvalResponse
		err  error
	}

	resultCh := make(chan result, len(c.Nodes))
	var wg sync.WaitGroup

	c.mu.Lock()
	weights := make([]float64, len(c.Model.Weights))
	copy(weights, c.Model.Weights)
	c.mu.Unlock()

	for _, node := range c.Nodes {
		if !node.Alive {
			continue
		}
		wg.Add(1)
		go func(n *NodeConn) {
			defer wg.Done()
			resp, err := n.requestEval(weights, timeout)
			resultCh <- result{resp: resp, err: err}
		}(node)
	}

	go func() {
		wg.Wait()
		close(resultCh)
	}()

	var tp, tn, fp, fn int
	for r := range resultCh {
		if r.err != nil {
			log.Println("nodo no respondio en evaluate:", r.err)
			continue
		}
		tp += r.resp.TP
		tn += r.resp.TN
		fp += r.resp.FP
		fn += r.resp.FN
	}

	total := tp + tn + fp + fn
	var accuracy, precision, recall, f1 float64
	if total > 0 {
		accuracy = float64(tp+tn) / float64(total) * 100
	}
	if tp+fp > 0 {
		precision = float64(tp) / float64(tp+fp) * 100
	}
	if tp+fn > 0 {
		recall = float64(tp) / float64(tp+fn) * 100
	}
	if precision+recall > 0 {
		f1 = 2 * (precision * recall) / (precision + recall)
	}

	return EvalMetrics{
		Accuracy: accuracy, Precision: precision, Recall: recall, F1: f1,
		TP: tp, TN: tn, FP: fp, FN: fn,
	}
}

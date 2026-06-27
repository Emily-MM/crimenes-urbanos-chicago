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
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	_ "api/docs"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/golang-jwt/jwt/v5"
	httpSwagger "github.com/swaggo/http-swagger"
	"go.mongodb.org/mongo-driver/v2/bson"
	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
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

var coordinator *Coordinator

type ApiMetrics struct {
	mu              sync.Mutex
	PredictCount    int64
	PredictTotalNs  int64
	LastTrainSecs   float64
	LastTrainEpochs int
	StartedAt       time.Time
}

var apiMetrics = &ApiMetrics{StartedAt: time.Now()}

type WSMessage struct {
	Type string `json:"type"`
	Data any    `json:"data"`
}

type WSHub struct {
	mu      sync.RWMutex
	clients map[*websocket.Conn]bool
}

var wsHub = &WSHub{clients: make(map[*websocket.Conn]bool)}

func (h *WSHub) Register(c *websocket.Conn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.clients[c] = true
}

func (h *WSHub) Unregister(c *websocket.Conn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.clients, c)
}

func (h *WSHub) Broadcast(msg WSMessage) {
	h.mu.RLock()
	conns := make([]*websocket.Conn, 0, len(h.clients))
	for c := range h.clients {
		conns = append(conns, c)
	}
	h.mu.RUnlock()

	for _, c := range conns {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		err := wsjson.Write(ctx, c, msg)
		cancel()
		if err != nil {
			h.Unregister(c)
		}
	}
}

func handleWS(w http.ResponseWriter, r *http.Request) {
	c, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		InsecureSkipVerify: true,
	})
	if err != nil {
		log.Printf("error aceptando conexion websocket: %v\n", err)
		return
	}
	defer c.CloseNow()

	wsHub.Register(c)
	defer wsHub.Unregister(c)
	log.Println("cliente websocket conectado")

	ctx := context.Background()
	for {
		_, _, err := c.Read(ctx)
		if err != nil {
			log.Println("cliente websocket desconectado")
			return
		}
	}
}

func startNodeStatusBroadcaster() {
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			if coordinator == nil {
				continue
			}
			var nodes []map[string]interface{}
			for _, n := range coordinator.Nodes {
				nodes = append(nodes, map[string]interface{}{
					"address": n.Address,
					"alive":   n.Alive,
				})
			}
			wsHub.Broadcast(WSMessage{
				Type: "node_status",
				Data: map[string]interface{}{
					"nodes":     nodes,
					"timestamp": time.Now().UTC(),
				},
			})
		}
	}()
}

var jwtSecret []byte

func loadJWTSecret() {
	secret := os.Getenv("JWT_SECRET")
	if secret == "" {
		secret = "pc4-dev-secret-cambiar-en-produccion"
		log.Println("aviso: JWT_SECRET no definido, usando secreto de desarrollo (no usar en produccion)")
	}
	jwtSecret = []byte(secret)
}

var authUser string
var authPassword string

func loadAuthCredentials() {
	authUser = os.Getenv("AUTH_USER")
	if authUser == "" {
		authUser = "admin"
	}
	authPassword = os.Getenv("AUTH_PASSWORD")
	if authPassword == "" {
		authPassword = "pc4admin"
		log.Println("aviso: AUTH_PASSWORD no definido, usando contrasena de desarrollo (no usar en produccion)")
	}
}

type LoginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

func generateToken(username string) (string, error) {
	claims := jwt.MapClaims{
		"sub": username,
		"exp": time.Now().Add(2 * time.Hour).Unix(),
		"iat": time.Now().Unix(),
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString(jwtSecret)
}

func validateToken(tokenStr string) (string, error) {
	token, err := jwt.Parse(tokenStr, func(t *jwt.Token) (interface{}, error) {
		if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("metodo de firma inesperado: %v", t.Header["alg"])
		}
		return jwtSecret, nil
	})
	if err != nil {
		return "", err
	}
	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok || !token.Valid {
		return "", errors.New("token invalido")
	}
	sub, _ := claims["sub"].(string)
	return sub, nil
}

// @Summary		Iniciar sesion
// @Description	Valida usuario y contrasena (definidos via AUTH_USER/AUTH_PASSWORD)
// @Description	y devuelve un JWT valido por 2 horas, necesario para /train y /ws.
// @Tags			autenticacion
// @Accept			json
// @Produce		json
// @Param			request	body		LoginRequest	true	"Usuario y contrasena"
// @Success		200		{object}	map[string]interface{}
// @Failure		401		{object}	map[string]interface{}
// @Router			/login [post]
func handleLogin(w http.ResponseWriter, r *http.Request) {
	var req LoginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "request invalido", http.StatusBadRequest)
		return
	}

	if req.Username != authUser || req.Password != authPassword {
		http.Error(w, "credenciales invalidas", http.StatusUnauthorized)
		return
	}

	token, err := generateToken(req.Username)
	if err != nil {
		http.Error(w, "error generando token", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"token":      token,
		"expires_in": 7200,
	})
}

func requireAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		header := r.Header.Get("Authorization")
		if header == "" || !strings.HasPrefix(header, "Bearer ") {
			http.Error(w, "falta header Authorization: Bearer <token>", http.StatusUnauthorized)
			return
		}
		tokenStr := strings.TrimPrefix(header, "Bearer ")

		if _, err := validateToken(tokenStr); err != nil {
			http.Error(w, "token invalido o expirado", http.StatusUnauthorized)
			return
		}
		next(w, r)
	}
}

func requireAuthQuery(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		tokenStr := r.URL.Query().Get("token")
		if tokenStr == "" {
			http.Error(w, "falta query param ?token=", http.StatusUnauthorized)
			return
		}
		if _, err := validateToken(tokenStr); err != nil {
			http.Error(w, "token invalido o expirado", http.StatusUnauthorized)
			return
		}
		next(w, r)
	}
}

type TrainingRecord struct {
	ID        bson.ObjectID `bson:"_id,omitempty"`
	Fecha     time.Time     `bson:"fecha"`
	Epochs    int           `bson:"epochs"`
	ElapsedS  float64       `bson:"elapsed_secs"`
	Loss      float64       `bson:"loss"`
	Accuracy  float64       `bson:"accuracy"`
	Precision float64       `bson:"precision"`
	Recall    float64       `bson:"recall"`
	F1        float64       `bson:"f1"`
	Weights   []float64     `bson:"weights"`
}

var mongoCollection *mongo.Collection
var predictionsCollection *mongo.Collection

func connectMongo() {
	uri := os.Getenv("MONGO_URI")
	if uri == "" {
		uri = "mongodb://localhost:27017"
	}
	dbName := os.Getenv("MONGO_DB")
	if dbName == "" {
		dbName = "chicago_crimes"
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	client, err := mongo.Connect(options.Client().ApplyURI(uri))
	if err != nil {
		log.Printf("aviso: no se pudo crear cliente Mongo: %v (el historial no se guardara)\n", err)
		return
	}

	if err := client.Ping(ctx, nil); err != nil {
		log.Printf("aviso: no se pudo conectar a Mongo en %s: %v (el historial no se guardara)\n", uri, err)
		return
	}

	mongoCollection = client.Database(dbName).Collection("training_history")
	predictionsCollection = client.Database(dbName).Collection("predictions")
	log.Printf("conectado a MongoDB en %s, base %s\n", uri, dbName)
}

func saveTrainingRecord(rec TrainingRecord) {
	if mongoCollection == nil {
		log.Println("aviso: Mongo no disponible, no se guarda el historial de este entrenamiento")
		return
	}
	rec.Fecha = time.Now()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if _, err := mongoCollection.InsertOne(ctx, rec); err != nil {
		log.Printf("error guardando historial en Mongo: %v\n", err)
	}
}

func loadLastTraining() (*TrainingRecord, bool) {
	if mongoCollection == nil {
		return nil, false
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	opts := options.FindOne().SetSort(bson.D{{Key: "_id", Value: -1}})
	var rec TrainingRecord
	err := mongoCollection.FindOne(ctx, bson.D{}, opts).Decode(&rec)
	if err != nil {
		if err != mongo.ErrNoDocuments {
			log.Printf("error leyendo ultimo entrenamiento de Mongo: %v\n", err)
		}
		return nil, false
	}
	return &rec, true
}

func loadAllTrainings(limit int) []TrainingRecord {
	if mongoCollection == nil {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	opts := options.Find().SetSort(bson.D{{Key: "_id", Value: -1}}).SetLimit(int64(limit))
	cursor, err := mongoCollection.Find(ctx, bson.D{}, opts)
	if err != nil {
		log.Printf("error leyendo historial de Mongo: %v\n", err)
		return nil
	}
	defer cursor.Close(ctx)

	var records []TrainingRecord
	if err := cursor.All(ctx, &records); err != nil {
		log.Printf("error decodificando historial de Mongo: %v\n", err)
		return nil
	}
	return records
}

type PredictionRecord struct {
	ID              bson.ObjectID `bson:"_id,omitempty"`
	Fecha           time.Time     `bson:"fecha"`
	Hour            int           `bson:"hour"`
	District        int           `bson:"district"`
	RiskProbability float64       `bson:"risk_probability"`
	RiskLevel       string        `bson:"risk_level"`
}

const dedupWindow = 60 * time.Second

func savePredictionIfNew(hour, district int, prob float64, level string) {
	if predictionsCollection == nil {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cutoff := time.Now().Add(-dedupWindow)
	filter := bson.D{
		{Key: "hour", Value: hour},
		{Key: "district", Value: district},
		{Key: "fecha", Value: bson.D{{Key: "$gte", Value: cutoff}}},
	}

	count, err := predictionsCollection.CountDocuments(ctx, filter)
	if err != nil {
		log.Printf("error verificando duplicados de prediccion en Mongo: %v\n", err)
		return
	}
	if count > 0 {
		return
	}

	rec := PredictionRecord{
		Fecha:           time.Now(),
		Hour:            hour,
		District:        district,
		RiskProbability: prob,
		RiskLevel:       level,
	}
	if _, err := predictionsCollection.InsertOne(ctx, rec); err != nil {
		log.Printf("error guardando prediccion en Mongo: %v\n", err)
	}
}

func loadRecentPredictions(limit int) []PredictionRecord {
	if predictionsCollection == nil {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	opts := options.Find().SetSort(bson.D{{Key: "_id", Value: -1}}).SetLimit(int64(limit))
	cursor, err := predictionsCollection.Find(ctx, bson.D{}, opts)
	if err != nil {
		log.Printf("error leyendo predicciones de Mongo: %v\n", err)
		return nil
	}
	defer cursor.Close(ctx)

	var records []PredictionRecord
	if err := cursor.All(ctx, &records); err != nil {
		log.Printf("error decodificando predicciones de Mongo: %v\n", err)
		return nil
	}
	return records
}

func (m *ApiMetrics) recordPredict(duration time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.PredictCount++
	m.PredictTotalNs += duration.Nanoseconds()
}

func (m *ApiMetrics) avgPredictMs() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.PredictCount == 0 {
		return 0
	}
	return float64(m.PredictTotalNs) / float64(m.PredictCount) / 1e6
}

// @Summary		Entrenar el modelo distribuido
// @Description	Dispara el entrenamiento del modelo en los 3 nodos del cluster via TCP.
// @Description	Reparte gradiente descendente paralelo y agrega resultados con resiliencia
// @Description	ante caida de nodos. Tambien evalua el modelo final automaticamente y
// @Description	guarda el resultado en MongoDB (coleccion training_history).
// @Description	Requiere JWT obtenido en /login.
// @Tags			entrenamiento
// @Produce		json
// @Security		BearerAuth
// @Success		200	{object}	map[string]interface{}
// @Failure		401	{object}	map[string]interface{}
// @Router			/train [post]
func handleTrain(w http.ResponseWriter, r *http.Request) {
	if coordinator == nil {
		http.Error(w, "coordinador no inicializado", http.StatusServiceUnavailable)
		return
	}

	epochs := 100
	lr := 0.3
	decay := 0.05

	start := time.Now()
	var lastLoss float64

	for epoch := 0; epoch < epochs; epoch++ {
		currentLR := lr / (1.0 + decay*float64(epoch))
		loss, nodosActivos := coordinator.trainEpoch(currentLR, 10*time.Second)
		lastLoss = loss
		if epoch%10 == 0 {
			log.Printf("epoca %d/%d  loss %.4f  nodos activos: %d\n", epoch+1, epochs, loss, nodosActivos)
		}

		wsHub.Broadcast(WSMessage{
			Type: "training_progress",
			Data: map[string]interface{}{
				"epoch":        epoch + 1,
				"total_epochs": epochs,
				"loss":         loss,
				"nodes_active": nodosActivos,
			},
		})
	}

	elapsed := time.Since(start)

	apiMetrics.mu.Lock()
	apiMetrics.LastTrainSecs = elapsed.Seconds()
	apiMetrics.LastTrainEpochs = epochs
	apiMetrics.mu.Unlock()

	log.Println("evaluando modelo final...")
	metrics := coordinator.evaluate(15 * time.Second)
	log.Printf("accuracy: %.2f%%  precision: %.2f%%  recall: %.2f%%  F1: %.2f\n",
		metrics.Accuracy, metrics.Precision, metrics.Recall, metrics.F1)

	weightsCopy := make([]float64, len(coordinator.Model.Weights))
	copy(weightsCopy, coordinator.Model.Weights)
	saveTrainingRecord(TrainingRecord{
		Epochs:    epochs,
		ElapsedS:  elapsed.Seconds(),
		Loss:      lastLoss,
		Accuracy:  metrics.Accuracy,
		Precision: metrics.Precision,
		Recall:    metrics.Recall,
		F1:        metrics.F1,
		Weights:   weightsCopy,
	})

	resp := map[string]interface{}{
		"status":       "completado",
		"epochs":       epochs,
		"final_loss":   lastLoss,
		"elapsed_secs": elapsed.Seconds(),
		"weights":      coordinator.Model.Weights,
		"metrics": map[string]interface{}{
			"accuracy":  metrics.Accuracy,
			"precision": metrics.Precision,
			"recall":    metrics.Recall,
			"f1":        metrics.F1,
			"tp":        metrics.TP,
			"tn":        metrics.TN,
			"fp":        metrics.FP,
			"fn":        metrics.FN,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func predictRisk(hour, district int) float64 {
	h := float64(hour) / 23.0
	d := float64(district) / 25.0
	esNoche := 0.0
	if hour >= 20 || hour <= 5 {
		esNoche = 1.0
	}
	esTarde := 0.0
	if hour >= 12 && hour < 20 {
		esTarde = 1.0
	}

	coordinator.mu.Lock()
	prob := coordinator.Model.Predict([]float64{h, d, 0.5, 0.5, 0.0, esNoche, esTarde, h * d, 0.8})
	coordinator.mu.Unlock()

	return prob
}

func nivelDe(prob float64) string {
	if prob >= 0.5 {
		return "alto"
	}
	return "bajo"
}

type PredictRequest struct {
	Hour     int `json:"hour"`
	District int `json:"district"`
}

// @Summary		Predecir riesgo de delito
// @Description	Dado una hora (0-23) y un distrito, devuelve la probabilidad
// @Description	de que sea una combinacion de alto riesgo segun el modelo entrenado.
// @Description	La consulta se guarda de forma asincrona en MongoDB (coleccion
// @Description	predictions), salvo que sea una repeticion de la misma consulta
// @Description	en los ultimos 60 segundos.
// @Tags			predicciones
// @Accept			json
// @Produce		json
// @Param			request	body		PredictRequest	true	"Hora y distrito a consultar"
// @Success		200		{object}	map[string]interface{}
// @Router			/predict [post]
func handlePredict(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	if coordinator == nil {
		http.Error(w, "coordinador no inicializado", http.StatusServiceUnavailable)
		return
	}

	var req PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "request invalido", http.StatusBadRequest)
		return
	}

	prob := predictRisk(req.Hour, req.District)
	level := nivelDe(prob)

	resp := map[string]interface{}{
		"hour":             req.Hour,
		"district":         req.District,
		"risk_probability": prob,
		"risk_level":       level,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)

	apiMetrics.recordPredict(time.Since(start))

	go savePredictionIfNew(req.Hour, req.District, prob, level)
}

var listaDistritos = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 22, 24, 25}

// @Summary		Mapa de calor por hora
// @Description	Devuelve la probabilidad de riesgo de los 22 distritos de Chicago
// @Description	para una hora especifica del dia. Pensado para alimentar un mapa de calor.
// @Tags			mapa de calor
// @Produce		json
// @Param			hour	query		int	true	"Hora del dia (0-23)"
// @Success		200		{object}	map[string]interface{}
// @Router			/heatmap [get]
func handleHeatmap(w http.ResponseWriter, r *http.Request) {
	if coordinator == nil {
		http.Error(w, "coordinador no inicializado", http.StatusServiceUnavailable)
		return
	}

	hourStr := r.URL.Query().Get("hour")
	hour, err := strconv.Atoi(hourStr)
	if err != nil || hour < 0 || hour > 23 {
		http.Error(w, "parametro 'hour' invalido, debe ser 0-23", http.StatusBadRequest)
		return
	}

	var puntos []map[string]interface{}
	for _, d := range listaDistritos {
		prob := predictRisk(hour, d)
		puntos = append(puntos, map[string]interface{}{
			"district":         d,
			"risk_probability": prob,
			"risk_level":       nivelDe(prob),
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"hour":   hour,
		"points": puntos,
	})
}

// @Summary		Mapa de calor completo
// @Description	Devuelve la matriz completa de riesgo para las 24 horas y los 22
// @Description	distritos en una sola llamada, para que el frontend la cargue una vez.
// @Tags			mapa de calor
// @Produce		json
// @Success		200	{object}	map[string]interface{}
// @Router			/heatmap/all [get]
func handleHeatmapAll(w http.ResponseWriter, r *http.Request) {
	if coordinator == nil {
		http.Error(w, "coordinador no inicializado", http.StatusServiceUnavailable)
		return
	}

	var matriz []map[string]interface{}
	for hour := 0; hour < 24; hour++ {
		var fila []map[string]interface{}
		for _, d := range listaDistritos {
			prob := predictRisk(hour, d)
			fila = append(fila, map[string]interface{}{
				"district":         d,
				"risk_probability": prob,
				"risk_level":       nivelDe(prob),
			})
		}
		matriz = append(matriz, map[string]interface{}{
			"hour":   hour,
			"points": fila,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"matrix": matriz})
}

// @Summary		Recomendacion de patrullaje
// @Description	Devuelve los N distritos de mayor riesgo para una hora especifica,
// @Description	ordenados de mayor a menor, con una recomendacion en texto.
// @Tags			recomendaciones
// @Produce		json
// @Param			hour	query		int	true	"Hora del dia (0-23)"
// @Param			limit	query		int	false	"Cantidad de distritos a recomendar (default 5)"
// @Success		200		{object}	map[string]interface{}
// @Router			/recommendation [get]
func handleRecommendation(w http.ResponseWriter, r *http.Request) {
	if coordinator == nil {
		http.Error(w, "coordinador no inicializado", http.StatusServiceUnavailable)
		return
	}

	hourStr := r.URL.Query().Get("hour")
	hour, err := strconv.Atoi(hourStr)
	if err != nil || hour < 0 || hour > 23 {
		http.Error(w, "parametro 'hour' invalido, debe ser 0-23", http.StatusBadRequest)
		return
	}

	limit := 5
	if l := r.URL.Query().Get("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil && parsed > 0 {
			limit = parsed
		}
	}

	type distritoRiesgo struct {
		District int
		Prob     float64
	}

	var resultados []distritoRiesgo
	for _, d := range listaDistritos {
		resultados = append(resultados, distritoRiesgo{District: d, Prob: predictRisk(hour, d)})
	}

	for i := 0; i < len(resultados); i++ {
		for j := i + 1; j < len(resultados); j++ {
			if resultados[j].Prob > resultados[i].Prob {
				resultados[i], resultados[j] = resultados[j], resultados[i]
			}
		}
	}

	if limit > len(resultados) {
		limit = len(resultados)
	}

	var prioridades []map[string]interface{}
	for i := 0; i < limit; i++ {
		prioridades = append(prioridades, map[string]interface{}{
			"district":         resultados[i].District,
			"risk_probability": resultados[i].Prob,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"hour":               hour,
		"recommendation":     fmt.Sprintf("a las %02d:00, priorizar patrullaje en %d distrito(s) de mayor riesgo", hour, limit),
		"priority_districts": prioridades,
	})
}

// @Summary		Evaluar el modelo
// @Description	Pide a cada nodo que evalue su particion local con los pesos
// @Description	actuales y consolida la matriz de confusion para calcular metricas globales.
// @Tags			entrenamiento
// @Produce		json
// @Success		200	{object}	map[string]interface{}
// @Router			/evaluate [get]
func handleEvaluate(w http.ResponseWriter, r *http.Request) {
	if coordinator == nil {
		http.Error(w, "coordinador no inicializado", http.StatusServiceUnavailable)
		return
	}
	metrics := coordinator.evaluate(15 * time.Second)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"accuracy":  metrics.Accuracy,
		"precision": metrics.Precision,
		"recall":    metrics.Recall,
		"f1":        metrics.F1,
		"tp":        metrics.TP,
		"tn":        metrics.TN,
		"fp":        metrics.FP,
		"fn":        metrics.FN,
	})
}

// @Summary		Chequeo de salud del sistema
// @Description	Verifica rapidamente cuantos nodos del cluster estan vivos,
// @Description	sin ejecutar ningun calculo pesado. Pensado para monitoreo automatico.
// @Tags			monitoreo
// @Produce		json
// @Success		200	{object}	map[string]interface{}
// @Failure		503	{object}	map[string]interface{}
// @Router			/health [get]
func handleHealth(w http.ResponseWriter, r *http.Request) {
	if coordinator == nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]string{"status": "coordinador no inicializado"})
		return
	}

	nodosVivos := 0
	for _, n := range coordinator.Nodes {
		if n.Alive {
			nodosVivos++
		}
	}

	status := "ok"
	code := http.StatusOK
	if nodosVivos == 0 {
		status = "degraded"
		code = http.StatusServiceUnavailable
	} else if nodosVivos < len(coordinator.Nodes) {
		status = "partial"
	}

	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":      status,
		"nodes_alive": nodosVivos,
		"nodes_total": len(coordinator.Nodes),
		"uptime_secs": time.Since(apiMetrics.StartedAt).Seconds(),
	})
}

// @Summary		Metricas de rendimiento
// @Description	Devuelve latencia promedio de /predict, duracion del ultimo
// @Description	entrenamiento, uptime de la API, y estado de cada nodo del cluster.
// @Tags			monitoreo
// @Produce		json
// @Success		200	{object}	map[string]interface{}
// @Router			/metrics [get]
func handleMetrics(w http.ResponseWriter, r *http.Request) {
	nodosVivos := 0
	var nodeDetails []map[string]interface{}
	if coordinator != nil {
		for _, n := range coordinator.Nodes {
			if n.Alive {
				nodosVivos++
			}
			nodeDetails = append(nodeDetails, map[string]interface{}{
				"address": n.Address,
				"alive":   n.Alive,
			})
		}
	}

	apiMetrics.mu.Lock()
	predictCount := apiMetrics.PredictCount
	lastTrainSecs := apiMetrics.LastTrainSecs
	lastTrainEpochs := apiMetrics.LastTrainEpochs
	apiMetrics.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"uptime_secs":             time.Since(apiMetrics.StartedAt).Seconds(),
		"predict_requests_total":  predictCount,
		"predict_avg_latency_ms":  apiMetrics.avgPredictMs(),
		"last_train_elapsed_secs": lastTrainSecs,
		"last_train_epochs":       lastTrainEpochs,
		"nodes_alive":             nodosVivos,
		"nodes_total":             len(nodeDetails),
		"nodes":                   nodeDetails,
	})
}

// @Summary		Estado de los nodos
// @Description	Devuelve la lista de nodos del cluster con su direccion TCP y si estan vivos.
// @Tags			monitoreo
// @Produce		json
// @Success		200	{object}	map[string]interface{}
// @Router			/status [get]
func handleStatus(w http.ResponseWriter, r *http.Request) {
	if coordinator == nil {
		http.Error(w, "coordinador no inicializado", http.StatusServiceUnavailable)
		return
	}
	var nodeStatus []map[string]interface{}
	for _, n := range coordinator.Nodes {
		nodeStatus = append(nodeStatus, map[string]interface{}{
			"address": n.Address,
			"alive":   n.Alive,
		})
	}
	json.NewEncoder(w).Encode(map[string]interface{}{"nodes": nodeStatus})
}

// @Summary		Historial de entrenamientos
// @Description	Devuelve los ultimos N entrenamientos guardados en MongoDB
// @Description	(coleccion training_history), del mas reciente al mas antiguo
// @Description	segun ObjectId. Si Mongo no esta disponible, devuelve lista vacia.
// @Tags			monitoreo
// @Produce		json
// @Param			limit	query		int	false	"Cantidad de registros a devolver (default 10)"
// @Success		200		{object}	map[string]interface{}
// @Router			/history [get]
func handleHistory(w http.ResponseWriter, r *http.Request) {
	limit := 10
	if l := r.URL.Query().Get("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil && parsed > 0 {
			limit = parsed
		}
	}

	records := loadAllTrainings(limit)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"count":   len(records),
		"history": records,
	})
}

// @Summary		Historial de predicciones
// @Description	Devuelve las ultimas N consultas a /predict guardadas en MongoDB
// @Description	(coleccion predictions), del mas reciente al mas antiguo.
// @Tags			predicciones
// @Produce		json
// @Param			limit	query		int	false	"Cantidad de registros a devolver (default 20)"
// @Success		200		{object}	map[string]interface{}
// @Router			/predictions/history [get]
func handlePredictionsHistory(w http.ResponseWriter, r *http.Request) {
	limit := 20
	if l := r.URL.Query().Get("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil && parsed > 0 {
			limit = parsed
		}
	}

	records := loadRecentPredictions(limit)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"count":       len(records),
		"predictions": records,
	})
}

func main() {
	dataPath := os.Getenv("DATA_PATH")
	if dataPath == "" {
		dataPath = "/datos/crimenes_limpio.csv"
	}

	nodeAddrs := []string{
		os.Getenv("NODO1_ADDR"),
		os.Getenv("NODO2_ADDR"),
		os.Getenv("NODO3_ADDR"),
	}

	loadJWTSecret()
	loadAuthCredentials()

	log.Println("conectando a MongoDB...")
	connectMongo()

	log.Println("cargando datos...")
	records, err := loadRecords(dataPath)
	if err != nil {
		log.Fatal("error cargando datos:", err)
	}
	log.Printf("%d registros cargados\n", len(records))

	altoRiesgo := calcularUmbrales(records)
	dataset := buildDataset(records, altoRiesgo)
	log.Printf("dataset construido: %d muestras\n", len(dataset))

	log.Println("conectando a nodos...")
	coordinator, err = NewCoordinator(nodeAddrs)
	if err != nil {
		log.Fatal("error creando coordinador:", err)
	}
	log.Printf("%d nodos conectados\n", len(coordinator.Nodes))

	if last, ok := loadLastTraining(); ok && len(last.Weights) == len(coordinator.Model.Weights) {
		coordinator.Model.Weights = last.Weights
		log.Printf("modelo previo cargado desde Mongo (entrenado el %s, accuracy %.2f%%)\n",
			last.Fecha.Format(time.RFC3339), last.Accuracy)
	} else {
		log.Println("no hay modelo previo en Mongo, se inicia desde pesos por defecto")
	}

	log.Println("repartiendo dataset a los nodos...")
	coordinator.initNodes(dataset)

	startNodeStatusBroadcaster()
	log.Println("broadcaster de estado de nodos iniciado (cada 5s via websocket)")

	http.HandleFunc("/login", handleLogin)
	http.HandleFunc("/train", requireAuth(handleTrain))
	http.HandleFunc("/predict", handlePredict)
	http.HandleFunc("/evaluate", handleEvaluate)
	http.HandleFunc("/heatmap", handleHeatmap)
	http.HandleFunc("/heatmap/all", handleHeatmapAll)
	http.HandleFunc("/recommendation", handleRecommendation)
	http.HandleFunc("/health", handleHealth)
	http.HandleFunc("/metrics", handleMetrics)
	http.HandleFunc("/status", handleStatus)
	http.HandleFunc("/history", handleHistory)
	http.HandleFunc("/predictions/history", handlePredictionsHistory)
	http.HandleFunc("/ws", requireAuthQuery(handleWS))

	http.Handle("/swagger/", httpSwagger.WrapHandler)

	log.Println("API escuchando en :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

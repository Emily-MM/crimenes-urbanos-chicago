FROM golang:1.22-alpine AS builder

WORKDIR /app

COPY go.mod ./
RUN go mod download

COPY main.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -o nodo main.go

FROM alpine:latest

WORKDIR /app
COPY --from=builder /app/nodo .

EXPOSE 9000

CMD ["./nodo"]

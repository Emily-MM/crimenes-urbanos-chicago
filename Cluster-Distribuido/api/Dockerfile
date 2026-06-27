FROM golang:1-alpine AS builder

WORKDIR /app

RUN apk add --no-cache git

RUN go install github.com/swaggo/swag/cmd/swag@v1.16.3

COPY go.mod ./
COPY main.go ./

RUN swag init -g main.go --output ./docs

RUN go mod tidy

RUN CGO_ENABLED=0 GOOS=linux go build -o api main.go

FROM alpine:latest

WORKDIR /app
COPY --from=builder /app/api .

EXPOSE 8080

CMD ["./api"]

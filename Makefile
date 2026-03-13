.PHONY: all build test bench lint clean deps examples

# Default target
all: deps lint test

# Install dependencies
deps:
	go mod download
	go mod tidy

# Run tests
test:
	go test -v -race ./...

# Run tests with coverage
cover:
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

# Run benchmarks
bench:
	cd bench && go test -bench=. -benchmem -benchtime=3s

# Run quick benchmarks
bench-quick:
	cd bench && go test -bench=. -benchmem -benchtime=1s

# Compare parallel vs single-threaded
bench-compare:
	cd bench && go test -bench='Worker' -benchmem -benchtime=3s

# Run linter
lint:
	@which golangci-lint > /dev/null || (echo "Installing golangci-lint..." && go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest)
	golangci-lint run ./...

# Format code
fmt:
	go fmt ./...
	goimports -w .

# Build examples
examples:
	go build -o bin/basic ./examples/basic
	go build -o bin/streaming ./examples/streaming

# Run basic example
run-basic: examples
	./bin/basic

# Run streaming example
run-streaming: examples
	./bin/streaming

# Clean build artifacts
clean:
	rm -rf bin/
	rm -f coverage.out coverage.html
	go clean -cache -testcache

# Check for vulnerabilities
vuln:
	@which govulncheck > /dev/null || go install golang.org/x/vuln/cmd/govulncheck@latest
	govulncheck ./...

# Generate documentation
docs:
	@echo "View docs at: https://pkg.go.dev/github.com/kraghavan/go-fast-token"
	godoc -http=:6060 &
	@echo "Local docs: http://localhost:6060/pkg/github.com/kraghavan/go-fast-token/"

# Prepare for release
release-check: deps lint test bench
	@echo "All checks passed!"
	@echo "To release:"
	@echo "  1. Update version in code if needed"
	@echo "  2. git tag v0.1.0"
	@echo "  3. git push origin v0.1.0"
	@echo "  4. pkg.go.dev will auto-index"

# Profile CPU
profile-cpu:
	cd bench && go test -bench=Encode_VeryLarge -cpuprofile=cpu.prof -benchtime=10s
	go tool pprof -http=:8080 bench/cpu.prof

# Profile memory
profile-mem:
	cd bench && go test -bench=Encode_VeryLarge -memprofile=mem.prof -benchtime=10s
	go tool pprof -http=:8080 bench/mem.prof

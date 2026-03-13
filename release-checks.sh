# Setup & test
cd go-fast-token
go mod tidy
go test ./...

# Benchmarks
cd bench
go test -bench=. -benchmem
go test -bench=. -benchmem -benchtime=3s   # More accurate

# Compare worker counts
go test -bench='Worker' -benchmem

# Profile CPU
go test -bench=Encode_VeryLarge -cpuprofile=cpu.prof
go tool pprof -http=:8080 cpu.prof

# Profile memory
go test -bench=Encode_VeryLarge -memprofile=mem.prof
go tool pprof -http=:8080 mem.prof
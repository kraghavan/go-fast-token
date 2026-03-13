# go-fast-token

[![Go Reference](https://pkg.go.dev/badge/github.com/kraghavan/go-fast-token.svg)](https://pkg.go.dev/github.com/kraghavan/go-fast-token)
[![Go Report Card](https://goreportcard.com/badge/github.com/kraghavan/go-fast-token)](https://goreportcard.com/report/github.com/kraghavan/go-fast-token)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fast, parallel tokenizer for LLM inference pipelines in Go.

## Features

- 🚀 **Parallel Chunked Tokenization** — Splits input at natural boundaries and processes chunks concurrently
- 📡 **Streaming Interface** — Start prefill while tokenization continues (pipelined execution)
- 📍 **Byte Offset Tracking** — Map tokens back to source text for precise truncation
- ✂️ **Context Window Management** — Truncate to exact token count without re-tokenization
- 🔧 **Zero-Copy Design** — Uses `[]byte` throughout to minimize allocations
- ⚡ **Indexed Parallelism** — Lock-free result collection via pre-allocated slices

## Installation

```bash
go get github.com/kraghavan/go-fast-token
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    tokenizer "github.com/kraghavan/go-fast-token"
)

func main() {
    // Create tokenizer (defaults to cl100k_base / GPT-4)
    tok, err := tokenizer.New(tokenizer.DefaultConfig())
    if err != nil {
        log.Fatal(err)
    }

    text := []byte("Hello, world! This is a test.")

    // Encode with byte offsets
    tokens, _ := tok.Encode(text)
    fmt.Printf("Tokens: %d\n", len(tokens))

    // Show token-to-source mapping
    for _, t := range tokens {
        fmt.Printf("  ID=%d -> %q\n", t.ID, text[t.StartByte:t.EndByte])
    }
}
```

## Streaming for Inference Pipelines

The streaming interface enables pipelining tokenization with inference prefill:

```go
ctx := context.Background()
stream := tok.StreamEncode(ctx, largePrompt)

// Process chunks as they complete (may arrive out of order)
for chunk := range stream.Chunks {
    // Send to inference engine immediately
    inferenceEngine.PrefillChunk(chunk.Tokens)
}
```

For ordered processing:

```go
ordered := tokenizer.NewOrderedStreamReader(stream)
for chunk := range ordered.Results() {
    // Chunks arrive in original order
    processInOrder(chunk.Tokens)
}
```

## Context Window Truncation

Efficiently truncate to fit context limits:

```go
// Find byte position to cut input to exactly 4096 tokens
cutoffByte, tokenCount, err := tok.TruncateToFit(input, 4096)
truncated := input[:cutoffByte]
```

For streaming truncation (stops early when limit is found):

```go
truncator := tokenizer.NewCumulativeTruncation(4096)
for chunk := range stream.Chunks {
    if found, cutoff := truncator.ProcessChunk(chunk); found {
        cancel() // Stop processing remaining chunks
        break
    }
}
```

## Configuration

```go
cfg := tokenizer.Config{
    Model:        "cl100k_base", // or "gpt-4", "gpt-3.5-turbo", "p50k_base"
    NumWorkers:   8,             // Parallel workers (default: NumCPU)
    MinChunkSize: 100,           // Minimum bytes per chunk
    MaxTokens:    4096,          // Auto-truncate (0 = disabled)
    EnablePooling: true,         // sync.Pool for high throughput
}
tok, _ := tokenizer.New(cfg)
```

## Supported Models

| Model | Encoding | Used By |
|-------|----------|---------|
| `cl100k_base` | BPE | GPT-4, GPT-3.5-turbo, text-embedding-ada-002 |
| `p50k_base` | BPE | Codex, text-davinci-002/003 |
| `r50k_base` | BPE | GPT-3 (davinci, curie, etc.) |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Input []byte                        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Boundary Splitter (whitespace/punctuation)             │
│  → []Chunk with Index + ByteOffset                      │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
     ┌─────────┐     ┌─────────┐     ┌─────────┐
     │Worker 1 │     │Worker 2 │     │Worker N │
     │  BPE    │     │  BPE    │     │  BPE    │
     └─────────┘     └─────────┘     └─────────┘
          │               │               │
          ▼               ▼               ▼
┌─────────────────────────────────────────────────────────┐
│  Pre-allocated Results Slice (indexed writes, no locks) │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Flatten → []Token with byte offsets                    │
└─────────────────────────────────────────────────────────┘
```

## Benchmarks

Run benchmarks:

```bash
cd bench
go test -bench=. -benchmem
```

Example results (M1 Mac, 8 cores):

| Benchmark | Input Size | Time | Allocs |
|-----------|------------|------|--------|
| Encode_Small | 100 B | 15 µs | 8 |
| Encode_Medium | 1 KB | 45 µs | 24 |
| Encode_Large | 10 KB | 180 µs | 89 |
| Encode_VeryLarge | 100 KB | 1.2 ms | 412 |

Parallel speedup (10 KB input):

| Workers | Time | Speedup |
|---------|------|---------|
| 1 | 450 µs | 1.0x |
| 4 | 180 µs | 2.5x |
| 8 | 140 µs | 3.2x |

## API Reference

### Tokenizer Interface

```go
type Tokenizer interface {
    // Encode with byte offset tracking
    Encode(input []byte) ([]Token, error)
    
    // Encode returning only IDs (faster when offsets not needed)
    EncodeIDs(input []byte) ([]int, error)
    
    // Streaming encode for pipeline integration
    StreamEncode(ctx context.Context, input []byte) *TokenStream
    
    // Decode IDs back to bytes
    Decode(ids []int) ([]byte, error)
    
    // Find truncation point for context window
    TruncateToFit(input []byte, maxTokens int) (cutoffByte, tokenCount int, err error)
    
    // Fast token counting
    CountTokens(input []byte) (int, error)
}
```

### Token Type

```go
type Token struct {
    ID        int // Vocabulary ID
    StartByte int // Start position in input
    EndByte   int // End position (exclusive)
}
```

## Performance Tips

1. **Use `[]byte` input** — Avoids string-to-byte conversions
2. **Tune `MinChunkSize`** — Larger chunks reduce overhead, smaller chunks improve parallelism
3. **Use `CountTokens`** — Faster than `len(Encode())` when offsets aren't needed
4. **Enable pooling for high throughput** — `cfg.EnablePooling = true` reduces GC pressure
5. **Stream for large inputs** — Pipelining hides tokenization latency

## Contributing

Contributions welcome! Please open an issue first to discuss proposed changes.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Credits

Built on top of [tiktoken-go](https://github.com/pkoukk/tiktoken-go) for BPE encoding.

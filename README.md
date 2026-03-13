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
- 🎯 **Auto-Tuned Workers** — Dynamically selects optimal worker count based on input size

![How the Fast-Token Factory Works](./assets/factory-infographic2.png)


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
    NumWorkers:   8,             // Max parallel workers (auto-tuned per request)
    MinChunkSize: 100,           // Minimum bytes per chunk
    MaxTokens:    4096,          // Auto-truncate (0 = disabled)
    EnablePooling: true,         // sync.Pool for high throughput
}
tok, _ := tokenizer.New(cfg)
```

> **Note:** `NumWorkers` sets the maximum — the tokenizer automatically selects fewer workers for smaller inputs to avoid coordination overhead.

## Supported Models

| Model | Encoding | Used By |
|-------|----------|---------|
| `o200k_base` | BPE | GPT-4o, GPT-4o-mini, o1, o3, o4-mini |
| `cl100k_base` | BPE | GPT-4, GPT-4-turbo, GPT-3.5-turbo, text-embedding-ada-002 |
| `p50k_base` | BPE | Codex, text-davinci-002/003 |
| `r50k_base` | BPE | GPT-3 (davinci, curie, etc.) |
```go
// GPT-4o (recommended for new projects)
cfg := tokenizer.DefaultConfig()
cfg.Model = "o200k_base"
tok, _ := tokenizer.New(cfg)

// Or use model name directly
tok, _ := tokenizer.NewWithModel("gpt-4o")

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
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Auto-Tune: optimalWorkers(inputSize, numChunks, max)   │
│  → Select 1-N workers based on workload                 │
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

<details>
<summary>Click to expand full architecture diagram</summary>

```mermaid
flowchart TB

    subgraph InputLayer["📥 INPUT LAYER"]
        direction TB
        INPUT[/"Raw Input []byte<br/>━━━━━━━━━━━━━━━<br/>• Zero-copy design<br/>• No string conversion<br/>• Preserves byte positions"/]
        CONFIG["⚙️ Config<br/>━━━━━━━━━━━━━━━<br/>• Model: cl100k_base<br/>• NumWorkers: runtime.NumCPU()<br/>• MinChunkSize: 100 bytes<br/>• MaxTokens: context limit<br/>• EnablePooling: sync.Pool"]
    end

    subgraph ChunkingLayer["✂️ BOUNDARY-AWARE CHUNKING"]
        direction TB
        SPLITTER["SplitByBoundary()<br/>━━━━━━━━━━━━━━━<br/>• Splits on whitespace/punctuation<br/>• Never cuts mid-word<br/>• Tracks ByteOffset per chunk<br/>• Merges micro-chunks"]
        
        subgraph Chunks["Generated Chunks"]
            direction LR
            CHUNK0["Chunk 0<br/>━━━━━━━<br/>Index: 0<br/>Data: []byte<br/>ByteOffset: 0"]
            CHUNK1["Chunk 1<br/>━━━━━━━<br/>Index: 1<br/>Data: []byte<br/>ByteOffset: 847"]
            CHUNK2["Chunk 2<br/>━━━━━━━<br/>Index: 2<br/>Data: []byte<br/>ByteOffset: 1694"]
            CHUNKN["Chunk N<br/>━━━━━━━<br/>Index: N<br/>Data: []byte<br/>ByteOffset: ..."]
        end
    end

    subgraph AutoTuneLayer["🎯 AUTO-TUNING"]
        direction TB
        AUTOTUNE["optimalWorkers()<br/>━━━━━━━━━━━━━━━<br/>• < 1KB → 1 worker<br/>• 1-5KB → 2 workers<br/>• 5-20KB → 4 workers<br/>• 20KB+ → scale up<br/>• Cap by chunk count"]
    end

    subgraph WorkerPoolLayer["⚡ PARALLEL WORKER POOL"]
        direction TB
        JOBCHAN[("Buffered Job Channel<br/>chan Chunk")]
        
        subgraph Workers["Goroutine Workers"]
            direction LR
            W1["🔧 Worker 1<br/>━━━━━━━━━<br/>• Pull from channel<br/>• BPE encode chunk<br/>• Track byte offsets<br/>• Write to results[idx]"]
            W2["🔧 Worker 2<br/>━━━━━━━━━<br/>• Pull from channel<br/>• BPE encode chunk<br/>• Track byte offsets<br/>• Write to results[idx]"]
            W3["🔧 Worker 3<br/>━━━━━━━━━<br/>• Pull from channel<br/>• BPE encode chunk<br/>• Track byte offsets<br/>• Write to results[idx]"]
            WN["🔧 Worker N<br/>━━━━━━━━━<br/>• Pull from channel<br/>• BPE encode chunk<br/>• Track byte offsets<br/>• Write to results[idx]"]
        end

        WAITGROUP["sync.WaitGroup<br/>━━━━━━━━━━━━━━━<br/>Synchronization barrier"]
    end

    subgraph BPELayer["🧠 BPE ENCODING (tiktoken-go)"]
        direction TB
        ENCODER["internal/bpe/Encoder<br/>━━━━━━━━━━━━━━━<br/>• Wraps tiktoken-go<br/>• Adds offset tracking<br/>• Optional sync.Pool<br/>• Supports cl100k, p50k, r50k"]
        
        subgraph TokenOutput["Token Structure"]
            TOKEN["Token{}<br/>━━━━━━━━━━━━━━━<br/>• ID: int (vocab ID)<br/>• StartByte: int<br/>• EndByte: int"]
        end
    end

    subgraph ResultsLayer["📦 LOCK-FREE RESULT COLLECTION"]
        direction TB
        PREALLOCATED["Pre-allocated Results Slice<br/>results := make([][]Token, numChunks)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>• Each worker owns its index<br/>• No mutex required<br/>• Direct indexed writes<br/>• Zero contention"]
        
        subgraph ResultSlots["Indexed Slots"]
            direction LR
            R0["results[0]<br/>[]Token"]
            R1["results[1]<br/>[]Token"]
            R2["results[2]<br/>[]Token"]
            RN["results[N]<br/>[]Token"]
        end

        FLATTEN["flattenResults()<br/>━━━━━━━━━━━━━━━<br/>Concatenate in order → []Token"]
    end

    subgraph OutputModes["📤 OUTPUT MODES"]
        direction TB
        
        subgraph BatchMode["Batch Mode"]
            ENCODE["Encode(input []byte) → []Token<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>• Blocks until complete<br/>• Returns ordered tokens<br/>• Full byte offset tracking"]
            ENCODEIDS["EncodeIDs(input []byte) → []int<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>• IDs only (faster)<br/>• No offset overhead"]
        end

        subgraph StreamMode["Streaming Mode"]
            STREAMENCODE["StreamEncode(ctx, input) → *TokenStream<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>• Returns immediately<br/>• Chunks arrive via channel<br/>• May be out of order<br/>• Context cancellation support"]
            TOKENSTREAM["TokenStream{}<br/>━━━━━━━━━━━━━━━<br/>• Chunks: chan ChunkResult<br/>• Done(): completion signal<br/>• Err(): error access<br/>• Wait(): blocking wait"]
        end
    end

    subgraph StreamUtilities["📡 STREAMING UTILITIES"]
        direction TB
        ORDERED["OrderedStreamReader<br/>━━━━━━━━━━━━━━━━━━━━<br/>• Buffers out-of-order chunks<br/>• Emits in sequence order<br/>• For ordered prefill"]
        COLLECTOR["StreamCollector<br/>━━━━━━━━━━━━━━━━━━━━<br/>• Thread-safe accumulator<br/>• Reorders on completion<br/>• Returns sorted []Token"]
        PREFILL["StreamToPrefill(callback)<br/>━━━━━━━━━━━━━━━━━━━━<br/>• Ordered chunk delivery<br/>• Direct inference integration<br/>• isFinal flag for last chunk"]
    end

    subgraph TruncationLayer["✂️ CONTEXT WINDOW MANAGEMENT"]
        direction TB
        TRUNCFIT["TruncateToFit(input, maxTokens)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>• Returns (cutoffByte, tokenCount)<br/>• Precise byte position<br/>• No re-tokenization needed"]
        CUMULATIVE["CumulativeTruncation{}<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>• ProcessChunk() → running count<br/>• Early exit when limit found<br/>• Streaming-compatible<br/>• Saves compute on long inputs"]
    end

    subgraph UseCases["🎯 USE CASES"]
        direction TB
        UC1["🚀 LLM Inference Prefill<br/>Pipeline tokenization with inference"]
        UC2["📊 Batch Processing<br/>High-throughput dataset tokenization"]
        UC3["🎯 RAG Chunking<br/>Precise document splitting"]
        UC4["📏 Context Management<br/>Fit prompts to model limits"]
    end

    INPUT --> SPLITTER
    CONFIG --> SPLITTER
    CONFIG --> AUTOTUNE
    
    SPLITTER --> CHUNK0 & CHUNK1 & CHUNK2 & CHUNKN
    
    CHUNK0 & CHUNK1 & CHUNK2 & CHUNKN --> AUTOTUNE
    AUTOTUNE --> JOBCHAN
    
    JOBCHAN --> W1 & W2 & W3 & WN
    
    W1 & W2 & W3 & WN --> ENCODER
    ENCODER --> TOKEN
    
    W1 -->|"results[0]"| R0
    W2 -->|"results[1]"| R1
    W3 -->|"results[2]"| R2
    WN -->|"results[N]"| RN
    
    W1 & W2 & W3 & WN --> WAITGROUP
    
    R0 & R1 & R2 & RN --> FLATTEN
    
    FLATTEN --> ENCODE
    FLATTEN --> ENCODEIDS
    
    JOBCHAN -.->|"Streaming path"| STREAMENCODE
    STREAMENCODE --> TOKENSTREAM
    
    TOKENSTREAM --> ORDERED
    TOKENSTREAM --> COLLECTOR
    ORDERED --> PREFILL
    
    ENCODE --> TRUNCFIT
    TOKENSTREAM --> CUMULATIVE
    
    ENCODE & STREAMENCODE --> UC1 & UC2 & UC3 & UC4

    classDef inputStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef chunkStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef autotuneStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef workerStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef bpeStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef resultStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef outputStyle fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef streamStyle fill:#fff8e1,stroke:#f9a825,stroke-width:2px
    classDef truncStyle fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef usecaseStyle fill:#f1f8e9,stroke:#558b2f,stroke-width:2px
    
    class INPUT,CONFIG inputStyle
    class SPLITTER,CHUNK0,CHUNK1,CHUNK2,CHUNKN chunkStyle
    class AUTOTUNE autotuneStyle
    class JOBCHAN,W1,W2,W3,WN,WAITGROUP workerStyle
    class ENCODER,TOKEN bpeStyle
    class PREALLOCATED,R0,R1,R2,RN,FLATTEN resultStyle
    class ENCODE,ENCODEIDS,STREAMENCODE,TOKENSTREAM outputStyle
    class ORDERED,COLLECTOR,PREFILL streamStyle
    class TRUNCFIT,CUMULATIVE truncStyle
    class UC1,UC2,UC3,UC4 usecaseStyle
```

</details>

## Benchmarks

Run benchmarks:

```bash
cd bench
go test -bench=. -benchmem
```

Results (Apple M4, 10 cores):

| Benchmark | Input Size | Time | Throughput |
|-----------|------------|------|------------|
| Encode_Small | 100 B | 11 µs | — |
| Encode_Medium | 1 KB | 126 µs | — |
| Encode_Large | 10 KB | 814 µs | — |
| Encode_VeryLarge | 100 KB | 11 ms | — |
| **Throughput** | 10 KB | 753 µs | **3.04M tokens/sec** |

### Auto-Tuning Impact

With auto-tuning, worker count is dynamically selected based on input size:

| Input Size | Workers Selected | Rationale |
|------------|------------------|-----------|
| < 1 KB | 1 | Parallelism overhead exceeds benefit |
| 1-5 KB | 2 | Light parallelism |
| 5-20 KB | 4 | Sweet spot |
| 20 KB+ | Scales up | More chunks = more workers |

### Before vs After Auto-Tuning

| Config | Before (fixed 8 workers) | After (auto-tuned) | Improvement |
|--------|--------------------------|---------------------|-------------|
| 10 KB input | 1,012 µs | 778 µs | **23% faster** |
| Throughput | 2.09M tok/s | 3.04M tok/s | **45% faster** |

> **Why?** Profiling with `GOGC=off` proved the slowdown was coordination overhead, not GC. Auto-tuning eliminates unnecessary goroutine scheduling.

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
2. **Trust auto-tuning** — Worker count adjusts automatically per request
3. **Use `CountTokens`** — Faster than `len(Encode())` when offsets aren't needed
4. **Enable pooling for high throughput** — `cfg.EnablePooling = true` reduces GC pressure
5. **Stream for large inputs** — Pipelining hides tokenization latency
6. **Profile before optimizing** — Run `GOGC=off` tests to isolate GC vs coordination overhead

## Contributing

Contributions welcome! Please open an issue first to discuss proposed changes.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Credits

Built on top of [tiktoken-go](https://github.com/pkoukk/tiktoken-go) for BPE encoding.


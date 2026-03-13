package tokenizer

import (
	"context"
	"runtime"
)

// Token represents a single token with byte offset tracking.
// Offsets are relative to the original input, enabling precise truncation
// and source mapping without re-tokenization.
type Token struct {
	ID        int // Token ID from vocabulary
	StartByte int // Start position in original input
	EndByte   int // End position in original input (exclusive)
}

// Chunk represents a unit of work for parallel processing.
// Uses []byte throughout to avoid string conversion overhead.
type Chunk struct {
	Index      int    // Position in original sequence (for reassembly)
	Data       []byte // Raw bytes to tokenize
	ByteOffset int    // Position in original input
}

// ChunkResult holds tokenization output from a worker.
type ChunkResult struct {
	Index  int     // Matches Chunk.Index for ordered reassembly
	Tokens []Token // Tokens with adjusted byte offsets
	Err    error   // Any error during tokenization
}

// TokenStream provides streaming access to tokenization results.
// Chunks may arrive out of order; use Index for reassembly.
type TokenStream struct {
	// Chunks delivers results as workers complete.
	// Closed when all work is done.
	Chunks <-chan ChunkResult

	// Internal
	done chan struct{}
	err  error
}

// Done returns a channel that closes when streaming is complete.
func (s *TokenStream) Done() <-chan struct{} {
	return s.done
}

// Err returns any error that occurred during streaming.
// Only valid after Done() is closed.
func (s *TokenStream) Err() error {
	return s.err
}

// Wait blocks until streaming completes and returns any error.
func (s *TokenStream) Wait() error {
	<-s.done
	return s.err
}

// Config controls tokenizer behavior.
type Config struct {
	// Model specifies the tokenizer model.
	// Supported: "gpt-4", "gpt-3.5-turbo", "cl100k_base", "p50k_base", "r50k_base"
	Model string

	// NumWorkers sets parallelism level.
	// Default: runtime.NumCPU()
	NumWorkers int

	// MinChunkSize prevents micro-chunks that add overhead.
	// Chunks smaller than this are merged with neighbors.
	// Default: 100 bytes
	MinChunkSize int

	// MaxTokens enables automatic truncation.
	// 0 = no limit (default)
	MaxTokens int

	// EnablePooling enables sync.Pool for buffer reuse.
	// Beneficial for high-throughput scenarios (>10K ops/sec).
	// Default: false
	EnablePooling bool
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		Model:        "cl100k_base", // GPT-4 / ChatGPT default
		NumWorkers:   runtime.NumCPU(),
		MinChunkSize: 100,
		MaxTokens:    0,
		EnablePooling: false,
	}
}

// Tokenizer is the main interface for tokenization operations.
type Tokenizer interface {
	// Encode tokenizes input bytes in parallel.
	// Returns tokens with byte offsets for source mapping.
	Encode(input []byte) ([]Token, error)

	// EncodeIDs is a convenience method returning only token IDs.
	EncodeIDs(input []byte) ([]int, error)

	// StreamEncode returns immediately with a stream of results.
	// Chunks may arrive out of order; use ChunkResult.Index for reassembly.
	// Cancellable via context.
	StreamEncode(ctx context.Context, input []byte) *TokenStream

	// Decode converts token IDs back to bytes.
	Decode(ids []int) ([]byte, error)

	// TruncateToFit finds the byte position to cut input to fit maxTokens.
	// Returns (cutoffByte, actualTokenCount, error).
	// Useful for context window management without re-tokenization.
	TruncateToFit(input []byte, maxTokens int) (cutoffByte int, tokenCount int, err error)

	// CountTokens returns the token count without full offset tracking.
	// Faster than len(Encode()) when offsets aren't needed.
	CountTokens(input []byte) (int, error)
}

// Stats holds performance metrics for benchmarking.
type Stats struct {
	InputBytes    int
	TokenCount    int
	ChunkCount    int
	WorkerCount   int
	TotalDuration int64 // nanoseconds
	ChunkDurations []int64 // per-chunk timing
}

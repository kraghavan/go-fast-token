package tokenizer

import (
	"context"
	"fmt"

	"github.com/kraghavan/go-fast-token/internal/bpe"
)

// tokenizer implements the Tokenizer interface with parallel processing.
type tokenizer struct {
	cfg     Config
	encoder *bpe.Encoder
}

// New creates a new parallel tokenizer with the given configuration.
func New(cfg Config) (Tokenizer, error) {
	// Apply defaults
	if cfg.Model == "" {
		cfg.Model = DefaultConfig().Model
	}
	if cfg.NumWorkers <= 0 {
		cfg.NumWorkers = DefaultConfig().NumWorkers
	}
	if cfg.MinChunkSize <= 0 {
		cfg.MinChunkSize = DefaultConfig().MinChunkSize
	}

	// Initialize BPE encoder
	encoder, err := bpe.NewEncoder(cfg.Model, cfg.EnablePooling)
	if err != nil {
		return nil, fmt.Errorf("failed to create encoder: %w", err)
	}

	return &tokenizer{
		cfg:     cfg,
		encoder: encoder,
	}, nil
}

// NewWithModel is a convenience constructor with just the model name.
func NewWithModel(model string) (Tokenizer, error) {
	cfg := DefaultConfig()
	cfg.Model = model
	return New(cfg)
}

// Encode tokenizes input bytes in parallel with byte offset tracking.
func (t *tokenizer) Encode(input []byte) ([]Token, error) {
	if len(input) == 0 {
		return nil, nil
	}

	// Small inputs: process directly without chunking overhead
	if len(input) <= t.cfg.MinChunkSize*2 {
		return t.encodeDirect(input)
	}

	// Split into chunks at natural boundaries
	chunks := SplitByBoundary(input, t.cfg.MinChunkSize)

	// Auto-tune worker count based on input size and chunks
	numWorkers := optimalWorkers(len(input), len(chunks), t.cfg.NumWorkers)

	// Create pool with optimal worker count
	pool := newWorkerPool(numWorkers, t.encoder)

	// Process in parallel
	ctx := context.Background()
	results, err := pool.processChunks(ctx, chunks)
	if err != nil {
		return nil, err
	}

	// Flatten results maintaining order
	tokens := flattenResults(results)

	// Apply truncation if configured
	if t.cfg.MaxTokens > 0 && len(tokens) > t.cfg.MaxTokens {
		tokens = tokens[:t.cfg.MaxTokens]
	}

	return tokens, nil
}

// encodeDirect handles small inputs without parallel overhead.
func (t *tokenizer) encodeDirect(input []byte) ([]Token, error) {
	bpeTokens, err := t.encoder.Encode(input)
	if err != nil {
		return nil, err
	}

	// Convert types
	tokens := make([]Token, len(bpeTokens))
	for i, bt := range bpeTokens {
		tokens[i] = Token{
			ID:        bt.ID,
			StartByte: bt.StartByte,
			EndByte:   bt.EndByte,
		}
	}

	return tokens, nil
}

// EncodeIDs returns only token IDs without offset tracking.
func (t *tokenizer) EncodeIDs(input []byte) ([]int, error) {
	if len(input) == 0 {
		return nil, nil
	}

	// Small inputs: direct path
	if len(input) <= t.cfg.MinChunkSize*2 {
		return t.encoder.EncodeIDs(input)
	}

	// Full encode then extract IDs
	tokens, err := t.Encode(input)
	if err != nil {
		return nil, err
	}

	ids := make([]int, len(tokens))
	for i, tok := range tokens {
		ids[i] = tok.ID
	}

	return ids, nil
}

// StreamEncode returns immediately with a stream of results.
// Useful for pipelining with inference engine prefill.
func (t *tokenizer) StreamEncode(ctx context.Context, input []byte) *TokenStream {
	done := make(chan struct{})

	// Handle empty input
	if len(input) == 0 {
		resultChan := make(chan ChunkResult)
		close(resultChan)
		close(done)
		return &TokenStream{
			Chunks: resultChan,
			done:   done,
		}
	}

	// Split into chunks
	chunks := SplitByBoundary(input, t.cfg.MinChunkSize)

	// Auto-tune worker count
	numWorkers := optimalWorkers(len(input), len(chunks), t.cfg.NumWorkers)

	// Create pool with optimal worker count
	pool := newWorkerPool(numWorkers, t.encoder)

	// Start streaming processing
	resultChan := pool.processChunksStreaming(ctx, chunks)

	// Create wrapper that tracks completion
	wrappedChan := make(chan ChunkResult, len(chunks))
	stream := &TokenStream{
		Chunks: wrappedChan,
		done:   done,
	}

	// Forward results and track completion
	go func() {
		defer close(done)
		defer close(wrappedChan)

		for result := range resultChan {
			if result.Err != nil {
				stream.err = result.Err
			}
			select {
			case wrappedChan <- result:
			case <-ctx.Done():
				stream.err = ctx.Err()
				return
			}
		}
	}()

	return stream
}

// Decode converts token IDs back to bytes.
func (t *tokenizer) Decode(ids []int) ([]byte, error) {
	return t.encoder.Decode(ids)
}

// TruncateToFit finds the byte position to cut input to fit maxTokens.
// Uses cumulative sum approach for efficiency (Gemini's suggestion).
func (t *tokenizer) TruncateToFit(input []byte, maxTokens int) (cutoffByte int, tokenCount int, err error) {
	if len(input) == 0 {
		return 0, 0, nil
	}

	if maxTokens <= 0 {
		return len(input), 0, fmt.Errorf("maxTokens must be positive")
	}

	// Tokenize in parallel
	tokens, err := t.Encode(input)
	if err != nil {
		return 0, 0, err
	}

	// Already fits
	if len(tokens) <= maxTokens {
		return len(input), len(tokens), nil
	}

	// Find cutoff point using cumulative approach
	// The token at index maxTokens-1 is the last one we keep
	lastToken := tokens[maxTokens-1]

	return lastToken.EndByte, maxTokens, nil
}

// CountTokens returns the token count without full offset tracking.
func (t *tokenizer) CountTokens(input []byte) (int, error) {
	if len(input) == 0 {
		return 0, nil
	}

	// For counting, direct path is usually faster
	return t.encoder.CountTokens(input), nil
}

// --- Additional utility methods ---

// EncodeWithContext allows cancellation of encoding.
func (t *tokenizer) EncodeWithContext(ctx context.Context, input []byte) ([]Token, error) {
	if len(input) == 0 {
		return nil, nil
	}

	// Check for early cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Small inputs: direct path
	if len(input) <= t.cfg.MinChunkSize*2 {
		return t.encodeDirect(input)
	}

	// Split and process with context
	chunks := SplitByBoundary(input, t.cfg.MinChunkSize)

	// Auto-tune worker count
	numWorkers := optimalWorkers(len(input), len(chunks), t.cfg.NumWorkers)
	pool := newWorkerPool(numWorkers, t.encoder)

	results, err := pool.processChunks(ctx, chunks)
	if err != nil {
		return nil, err
	}

	return flattenResults(results), nil
}

// CollectStream gathers all streaming results into ordered tokens.
// Convenience method that handles reassembly.
func CollectStream(ctx context.Context, stream *TokenStream) ([]Token, error) {
	var allResults []ChunkResult

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case result, ok := <-stream.Chunks:
			if !ok {
				// Channel closed, assemble results
				return assembleResults(allResults)
			}
			if result.Err != nil {
				return nil, result.Err
			}
			allResults = append(allResults, result)
		}
	}
}

// assembleResults sorts and flattens chunk results.
func assembleResults(results []ChunkResult) ([]Token, error) {
	if len(results) == 0 {
		return nil, nil
	}

	// Find max index to size the slice
	maxIndex := 0
	for _, r := range results {
		if r.Index > maxIndex {
			maxIndex = r.Index
		}
	}

	// Place results by index
	ordered := make([][]Token, maxIndex+1)
	for _, r := range results {
		ordered[r.Index] = r.Tokens
	}

	// Flatten
	return flattenResults(ordered), nil
}

// GetConfig returns the current configuration.
func (t *tokenizer) GetConfig() Config {
	return t.cfg
}

// optimalWorkers calculates the best worker count based on input size and chunks.
// Based on benchmark findings: 4 workers is optimal for 10KB, more workers adds overhead.
func optimalWorkers(inputSize int, numChunks int, maxWorkers int) int {
	// Heuristic based on benchmarks:
	// - Below 1KB: single-threaded is faster (no coordination overhead)
	// - 1-5KB: 2 workers
	// - 5-20KB: 4 workers (sweet spot)
	// - 20KB+: scale up, but cap at maxWorkers

	if inputSize < 1024 {
		return 1
	}
	if inputSize < 5*1024 {
		return minInt(2, maxWorkers)
	}
	if inputSize < 20*1024 {
		return minInt(4, maxWorkers)
	}

	// For larger inputs, scale with size but cap
	// Roughly 1 worker per 5KB, max out at configured limit
	workers := inputSize / (5 * 1024)
	if workers < 4 {
		workers = 4
	}
	if workers > maxWorkers {
		workers = maxWorkers
	}

	// Also cap by chunk count — no point having more workers than chunks
	if workers > numChunks {
		workers = numChunks
	}

	return workers
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

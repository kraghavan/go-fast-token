package tokenizer

import (
	"context"
	"sort"
	"sync"
)

// StreamCollector buffers and reorders streaming results.
// Useful when you need ordered output from a streaming tokenizer.
type StreamCollector struct {
	mu       sync.Mutex
	results  map[int]ChunkResult
	expected int
	received int
	done     bool
	err      error
}

// NewStreamCollector creates a collector for the expected number of chunks.
func NewStreamCollector(expectedChunks int) *StreamCollector {
	return &StreamCollector{
		results:  make(map[int]ChunkResult, expectedChunks),
		expected: expectedChunks,
	}
}

// Add stores a chunk result. Thread-safe.
func (c *StreamCollector) Add(result ChunkResult) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if result.Err != nil && c.err == nil {
		c.err = result.Err
	}

	c.results[result.Index] = result
	c.received++

	if c.received >= c.expected {
		c.done = true
	}
}

// IsDone returns true when all expected chunks are received.
func (c *StreamCollector) IsDone() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.done
}

// Err returns any error encountered.
func (c *StreamCollector) Err() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.err
}

// Tokens returns all tokens in order. Only call after IsDone() returns true.
func (c *StreamCollector) Tokens() []Token {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Sort by index
	indices := make([]int, 0, len(c.results))
	for idx := range c.results {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	// Flatten in order
	var tokens []Token
	for _, idx := range indices {
		tokens = append(tokens, c.results[idx].Tokens...)
	}

	return tokens
}

// OrderedStreamReader wraps a stream to deliver results in order.
// Buffers out-of-order results until earlier chunks arrive.
type OrderedStreamReader struct {
	stream   *TokenStream
	buffer   map[int]ChunkResult
	nextIdx  int
	outChan  chan ChunkResult
	doneChan chan struct{}
	err      error
}

// NewOrderedStreamReader wraps a stream for ordered delivery.
func NewOrderedStreamReader(stream *TokenStream) *OrderedStreamReader {
	r := &OrderedStreamReader{
		stream:   stream,
		buffer:   make(map[int]ChunkResult),
		nextIdx:  0,
		outChan:  make(chan ChunkResult, 16),
		doneChan: make(chan struct{}),
	}

	go r.run()
	return r
}

func (r *OrderedStreamReader) run() {
	defer close(r.doneChan)
	defer close(r.outChan)

	for result := range r.stream.Chunks {
		if result.Err != nil {
			r.err = result.Err
			r.outChan <- result
			return
		}

		// Buffer if not next expected
		if result.Index != r.nextIdx {
			r.buffer[result.Index] = result
			continue
		}

		// Emit this one
		r.outChan <- result
		r.nextIdx++

		// Emit any buffered consecutive chunks
		for {
			buffered, ok := r.buffer[r.nextIdx]
			if !ok {
				break
			}
			delete(r.buffer, r.nextIdx)
			r.outChan <- buffered
			r.nextIdx++
		}
	}

	// Emit any remaining buffered (shouldn't happen in normal operation)
	for r.nextIdx < r.nextIdx+len(r.buffer) {
		if result, ok := r.buffer[r.nextIdx]; ok {
			r.outChan <- result
			delete(r.buffer, r.nextIdx)
		}
		r.nextIdx++
	}
}

// Results returns the ordered output channel.
func (r *OrderedStreamReader) Results() <-chan ChunkResult {
	return r.outChan
}

// Done returns a channel that closes when streaming is complete.
func (r *OrderedStreamReader) Done() <-chan struct{} {
	return r.doneChan
}

// Err returns any error encountered.
func (r *OrderedStreamReader) Err() error {
	return r.err
}

// PrefillCallback is called for each chunk as it becomes ready.
// Return an error to abort processing.
type PrefillCallback func(tokens []Token, chunkIndex int, isFinal bool) error

// StreamToPrefill connects tokenizer output to an inference prefill stage.
// Calls the callback for each chunk in order.
func StreamToPrefill(ctx context.Context, stream *TokenStream, numChunks int, callback PrefillCallback) error {
	ordered := NewOrderedStreamReader(stream)

	chunksProcessed := 0
	for result := range ordered.Results() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if result.Err != nil {
			return result.Err
		}

		chunksProcessed++
		isFinal := chunksProcessed >= numChunks

		if err := callback(result.Tokens, result.Index, isFinal); err != nil {
			return err
		}
	}

	return ordered.Err()
}

// CumulativeTruncation calculates where to cut based on streaming results.
// Returns as soon as it determines the cutoff, without waiting for all chunks.
type CumulativeTruncation struct {
	maxTokens    int
	tokensSoFar  int
	cutoffFound  bool
	cutoffByte   int
	cutoffTokens int
	mu           sync.Mutex
}

// NewCumulativeTruncation creates a truncation calculator.
func NewCumulativeTruncation(maxTokens int) *CumulativeTruncation {
	return &CumulativeTruncation{
		maxTokens: maxTokens,
	}
}

// ProcessChunk adds a chunk's tokens to the running count.
// Returns true if the cutoff point is within this chunk.
func (ct *CumulativeTruncation) ProcessChunk(result ChunkResult) (foundCutoff bool, cutoffByte int) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	if ct.cutoffFound {
		return false, 0
	}

	for _, token := range result.Tokens {
		ct.tokensSoFar++
		if ct.tokensSoFar >= ct.maxTokens {
			ct.cutoffFound = true
			ct.cutoffByte = token.EndByte
			ct.cutoffTokens = ct.tokensSoFar
			return true, token.EndByte
		}
	}

	return false, 0
}

// IsCutoffFound returns whether the cutoff has been determined.
func (ct *CumulativeTruncation) IsCutoffFound() bool {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	return ct.cutoffFound
}

// Result returns the cutoff position. Only valid after IsCutoffFound() returns true.
func (ct *CumulativeTruncation) Result() (cutoffByte int, tokenCount int) {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	return ct.cutoffByte, ct.cutoffTokens
}

// TokenCount returns the current running token count.
func (ct *CumulativeTruncation) TokenCount() int {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	return ct.tokensSoFar
}

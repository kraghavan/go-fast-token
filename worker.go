package tokenizer

import (
	"context"
	"sync"

	"github.com/kraghavan/go-fast-token/internal/bpe"
)

// workerPool manages parallel tokenization with indexed result collection.
// Uses pre-allocated result slice - each worker writes to its own index,
// eliminating the need for mutex locks (Gemini tip #3).
type workerPool struct {
	numWorkers int
	encoder    *bpe.Encoder
}

// newWorkerPool creates a pool with the specified number of workers.
func newWorkerPool(numWorkers int, encoder *bpe.Encoder) *workerPool {
	return &workerPool{
		numWorkers: numWorkers,
		encoder:    encoder,
	}
}

// processChunks tokenizes all chunks in parallel and returns ordered results.
// This is the batch (non-streaming) mode.
func (wp *workerPool) processChunks(ctx context.Context, chunks []Chunk) ([][]Token, error) {
	if len(chunks) == 0 {
		return nil, nil
	}

	// Pre-allocate results slice - Gemini tip #2
	// Each worker writes directly to its index - no locking needed
	results := make([][]Token, len(chunks))
	errors := make([]error, len(chunks))

	// Job channel
	jobs := make(chan Chunk, len(chunks))

	// WaitGroup for synchronization
	var wg sync.WaitGroup

	// Spawn workers
	numWorkers := wp.numWorkers
	if numWorkers > len(chunks) {
		numWorkers = len(chunks) // Don't spawn more workers than jobs
	}

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			wp.worker(ctx, jobs, results, errors)
		}()
	}

	// Send jobs
	for _, chunk := range chunks {
		select {
		case jobs <- chunk:
		case <-ctx.Done():
			close(jobs)
			return nil, ctx.Err()
		}
	}
	close(jobs)

	// Wait for completion
	wg.Wait()

	// Check for errors
	for i, err := range errors {
		if err != nil {
			return nil, &ChunkError{Index: i, Err: err}
		}
	}

	return results, nil
}

// worker processes chunks from the jobs channel.
// Writes results directly to pre-allocated slice at chunk.Index.
func (wp *workerPool) worker(
	ctx context.Context,
	jobs <-chan Chunk,
	results [][]Token,
	errors []error,
) {
	for {
		select {
		case <-ctx.Done():
			return
		case chunk, ok := <-jobs:
			if !ok {
				return // Channel closed
			}

			// Tokenize with base offset adjustment
			tokens, err := wp.encoder.EncodeWithBaseOffset(chunk.Data, chunk.ByteOffset)
			if err != nil {
				errors[chunk.Index] = err
				continue
			}

			// Convert internal Token type to package Token type
			pkgTokens := make([]Token, len(tokens))
			for i, t := range tokens {
				pkgTokens[i] = Token{
					ID:        t.ID,
					StartByte: t.StartByte,
					EndByte:   t.EndByte,
				}
			}

			// Direct indexed write - no mutex needed (Gemini tip #3)
			results[chunk.Index] = pkgTokens
		}
	}
}

// processChunksStreaming tokenizes chunks and sends results as they complete.
// Results may arrive out of order; use ChunkResult.Index for reassembly.
func (wp *workerPool) processChunksStreaming(
	ctx context.Context,
	chunks []Chunk,
) <-chan ChunkResult {
	resultChan := make(chan ChunkResult, len(chunks))

	if len(chunks) == 0 {
		close(resultChan)
		return resultChan
	}

	go func() {
		defer close(resultChan)

		// Semaphore to limit concurrent workers
		sem := make(chan struct{}, wp.numWorkers)
		var wg sync.WaitGroup

		for _, chunk := range chunks {
			// Check for cancellation
			select {
			case <-ctx.Done():
				resultChan <- ChunkResult{Index: chunk.Index, Err: ctx.Err()}
				return
			default:
			}

			// Acquire semaphore
			sem <- struct{}{}
			wg.Add(1)

			go func(c Chunk) {
				defer func() {
					<-sem // Release semaphore
					wg.Done()
				}()

				tokens, err := wp.encoder.EncodeWithBaseOffset(c.Data, c.ByteOffset)

				// Convert types
				var pkgTokens []Token
				if err == nil {
					pkgTokens = make([]Token, len(tokens))
					for i, t := range tokens {
						pkgTokens[i] = Token{
							ID:        t.ID,
							StartByte: t.StartByte,
							EndByte:   t.EndByte,
						}
					}
				}

				// Send result (may block if consumer is slow)
				select {
				case resultChan <- ChunkResult{
					Index:  c.Index,
					Tokens: pkgTokens,
					Err:    err,
				}:
				case <-ctx.Done():
					return
				}
			}(chunk)
		}

		wg.Wait()
	}()

	return resultChan
}

// ChunkError wraps an error with the chunk index that caused it.
type ChunkError struct {
	Index int
	Err   error
}

func (e *ChunkError) Error() string {
	return e.Err.Error()
}

func (e *ChunkError) Unwrap() error {
	return e.Err
}

// flattenResults combines chunked results into a single token slice.
// Preserves order based on chunk index.
func flattenResults(results [][]Token) []Token {
	// Count total tokens for pre-allocation
	total := 0
	for _, r := range results {
		total += len(r)
	}

	// Pre-allocate - Gemini tip #2
	tokens := make([]Token, 0, total)

	for _, r := range results {
		tokens = append(tokens, r...)
	}

	return tokens
}

// flattenIDs extracts only token IDs from results.
func flattenIDs(results [][]Token) []int {
	total := 0
	for _, r := range results {
		total += len(r)
	}

	ids := make([]int, 0, total)
	for _, r := range results {
		for _, t := range r {
			ids = append(ids, t.ID)
		}
	}

	return ids
}

// collectStreamResults gathers streaming results and returns them in order.
// Blocks until all chunks are received or context is cancelled.
func collectStreamResults(ctx context.Context, stream <-chan ChunkResult, numChunks int) ([][]Token, error) {
	// Pre-allocate results slice
	results := make([][]Token, numChunks)
	received := 0

	for received < numChunks {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case result, ok := <-stream:
			if !ok {
				// Channel closed early
				if received < numChunks {
					return nil, &ChunkError{Index: received, Err: context.Canceled}
				}
				break
			}

			if result.Err != nil {
				return nil, &ChunkError{Index: result.Index, Err: result.Err}
			}

			results[result.Index] = result.Tokens
			received++
		}
	}

	return results, nil
}

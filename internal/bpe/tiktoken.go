package bpe

import (
	"fmt"
	"sync"

	"github.com/pkoukk/tiktoken-go"
)

// Encoder wraps tiktoken-go with offset tracking capabilities.
type Encoder struct {
	tk        *tiktoken.Tiktoken
	model     string
	
	// Buffer pool for high-throughput scenarios
	poolEnabled bool
	bufPool     *sync.Pool
}

// Token represents a token with its position in the source.
type Token struct {
	ID        int
	StartByte int
	EndByte   int
}

// NewEncoder creates a BPE encoder for the specified model.
// Supported models: "gpt-4", "gpt-3.5-turbo", "cl100k_base", "p50k_base", "r50k_base"
func NewEncoder(model string, enablePool bool) (*Encoder, error) {
	tk, err := tiktoken.EncodingForModel(model)
	if err != nil {
		// Try as encoding name directly
		tk, err = tiktoken.GetEncoding(model)
		if err != nil {
			return nil, fmt.Errorf("failed to load tokenizer for %q: %w", model, err)
		}
	}

	enc := &Encoder{
		tk:          tk,
		model:       model,
		poolEnabled: enablePool,
	}

	if enablePool {
		enc.bufPool = &sync.Pool{
			New: func() interface{} {
				buf := make([]int, 0, 1024)
				return &buf
			},
		}
	}

	return enc, nil
}

// Encode tokenizes input and returns token IDs with byte offsets.
// The offsets are relative to the start of the input slice.
func (e *Encoder) Encode(input []byte) ([]Token, error) {
	if len(input) == 0 {
		return nil, nil
	}

	// Get raw token IDs from tiktoken
	// tiktoken-go works with strings, so we must convert here
	// (this is unavoidable without reimplementing BPE)
	ids := e.tk.Encode(string(input), nil, nil)

	if len(ids) == 0 {
		return nil, nil
	}

	// Build tokens with offset tracking
	tokens := make([]Token, 0, len(ids))
	bytePos := 0

	for _, id := range ids {
		// Decode this single token to get its text
		tokenText := e.tk.Decode([]int{id})
		tokenBytes := len(tokenText)

		tokens = append(tokens, Token{
			ID:        id,
			StartByte: bytePos,
			EndByte:   bytePos + tokenBytes,
		})

		bytePos += tokenBytes
	}

	return tokens, nil
}

// EncodeIDs returns only token IDs without offset tracking.
// Faster when offsets aren't needed.
func (e *Encoder) EncodeIDs(input []byte) ([]int, error) {
	if len(input) == 0 {
		return nil, nil
	}

	ids := e.tk.Encode(string(input), nil, nil)
	return ids, nil
}

// EncodeWithBaseOffset tokenizes and adjusts offsets by baseOffset.
// Used for chunk processing where chunk starts at a position in the original input.
func (e *Encoder) EncodeWithBaseOffset(input []byte, baseOffset int) ([]Token, error) {
	tokens, err := e.Encode(input)
	if err != nil {
		return nil, err
	}

	// Adjust offsets
	for i := range tokens {
		tokens[i].StartByte += baseOffset
		tokens[i].EndByte += baseOffset
	}

	return tokens, nil
}

// Decode converts token IDs back to bytes.
func (e *Encoder) Decode(ids []int) ([]byte, error) {
	if len(ids) == 0 {
		return nil, nil
	}

	text := e.tk.Decode(ids)
	return []byte(text), nil
}

// CountTokens returns the number of tokens without full encoding.
func (e *Encoder) CountTokens(input []byte) int {
	if len(input) == 0 {
		return 0
	}
	ids := e.tk.Encode(string(input), nil, nil)
	return len(ids)
}

// Model returns the model name this encoder was created for.
func (e *Encoder) Model() string {
	return e.model
}

// SpecialTokens returns the special token mappings if available.
func (e *Encoder) SpecialTokens() map[string]int {
	// tiktoken-go doesn't expose this directly
	// Common special tokens for cl100k_base:
	return map[string]int{
		"<|endoftext|>":   100257,
		"<|fim_prefix|>":  100258,
		"<|fim_middle|>":  100259,
		"<|fim_suffix|>":  100260,
		"<|endofprompt|>": 100276,
	}
}

// --- Buffer pool methods for high-throughput scenarios ---

// getBuffer retrieves a buffer from the pool.
func (e *Encoder) getBuffer() *[]int {
	if !e.poolEnabled || e.bufPool == nil {
		buf := make([]int, 0, 1024)
		return &buf
	}
	return e.bufPool.Get().(*[]int)
}

// putBuffer returns a buffer to the pool.
func (e *Encoder) putBuffer(buf *[]int) {
	if !e.poolEnabled || e.bufPool == nil {
		return
	}
	*buf = (*buf)[:0] // Reset length, keep capacity
	e.bufPool.Put(buf)
}

// EncodeBatch tokenizes multiple inputs efficiently.
// Useful for batch processing scenarios.
func (e *Encoder) EncodeBatch(inputs [][]byte) ([][]Token, error) {
	results := make([][]Token, len(inputs))
	
	for i, input := range inputs {
		tokens, err := e.Encode(input)
		if err != nil {
			return nil, fmt.Errorf("failed to encode input %d: %w", i, err)
		}
		results[i] = tokens
	}
	
	return results, nil
}

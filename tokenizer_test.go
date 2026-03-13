package tokenizer

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestSplitByBoundary(t *testing.T) {
	tests := []struct {
		name         string
		input        string
		minChunkSize int
		wantChunks   int
	}{
		{
			name:         "empty input",
			input:        "",
			minChunkSize: 100,
			wantChunks:   0,
		},
		{
			name:         "small input no split",
			input:        "hello world",
			minChunkSize: 100,
			wantChunks:   1,
		},
		{
			name:         "split on whitespace",
			input:        "The quick brown fox jumps over the lazy dog",
			minChunkSize: 10,
			wantChunks:   4, // Approximately
		},
		{
			name:         "long words",
			input:        "supercalifragilisticexpialidocious is a word",
			minChunkSize: 10,
			wantChunks:   2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks := SplitByBoundary([]byte(tt.input), tt.minChunkSize)

			if len(chunks) < tt.wantChunks-1 || len(chunks) > tt.wantChunks+1 {
				t.Errorf("SplitByBoundary() got %d chunks, want approximately %d", len(chunks), tt.wantChunks)
			}

			// Verify indices are sequential
			for i, chunk := range chunks {
				if chunk.Index != i {
					t.Errorf("Chunk %d has index %d, want %d", i, chunk.Index, i)
				}
			}

			// Verify no data loss
			var reconstructed []byte
			for _, chunk := range chunks {
				reconstructed = append(reconstructed, chunk.Data...)
			}

			// Allow for whitespace trimming
			originalTrimmed := strings.TrimSpace(tt.input)
			reconstructedTrimmed := strings.TrimSpace(string(reconstructed))

			if len(originalTrimmed) > 0 && !strings.Contains(reconstructedTrimmed, originalTrimmed[:min(10, len(originalTrimmed))]) {
				t.Errorf("Data loss in split: original=%q, reconstructed=%q", tt.input, string(reconstructed))
			}
		})
	}
}

func TestSplitByBoundaryOffsets(t *testing.T) {
	input := "Hello world, this is a test."
	chunks := SplitByBoundary([]byte(input), 5)

	// Verify byte offsets are reasonable
	for i, chunk := range chunks {
		if chunk.ByteOffset < 0 || chunk.ByteOffset > len(input) {
			t.Errorf("Chunk %d has invalid ByteOffset: %d", i, chunk.ByteOffset)
		}

		// Verify the data at offset matches
		if chunk.ByteOffset+len(chunk.Data) <= len(input) {
			expected := input[chunk.ByteOffset : chunk.ByteOffset+len(chunk.Data)]
			if string(chunk.Data) != expected {
				t.Errorf("Chunk %d data mismatch at offset %d: got %q, want %q",
					i, chunk.ByteOffset, string(chunk.Data), expected)
			}
		}
	}
}

func TestMergeSmallChunks(t *testing.T) {
	chunks := []Chunk{
		{Index: 0, Data: []byte("a"), ByteOffset: 0},
		{Index: 1, Data: []byte("b"), ByteOffset: 1},
		{Index: 2, Data: []byte("c"), ByteOffset: 2},
		{Index: 3, Data: []byte("longer text"), ByteOffset: 3},
	}

	merged := MergeSmallChunks(chunks, 10)

	if len(merged) >= len(chunks) {
		t.Errorf("Expected fewer chunks after merge, got %d from %d", len(merged), len(chunks))
	}

	// Verify indices are sequential after merge
	for i, chunk := range merged {
		if chunk.Index != i {
			t.Errorf("Merged chunk %d has index %d, want %d", i, chunk.Index, i)
		}
	}
}

func TestTokenizerEncode(t *testing.T) {
	tok, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	tests := []struct {
		name      string
		input     string
		wantEmpty bool
	}{
		{
			name:      "empty input",
			input:     "",
			wantEmpty: true,
		},
		{
			name:      "simple text",
			input:     "Hello, world!",
			wantEmpty: false,
		},
		{
			name:      "longer text",
			input:     "The quick brown fox jumps over the lazy dog. " + strings.Repeat("This is additional text. ", 20),
			wantEmpty: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens, err := tok.Encode([]byte(tt.input))
			if err != nil {
				t.Fatalf("Encode() error = %v", err)
			}

			if tt.wantEmpty && len(tokens) != 0 {
				t.Errorf("Expected empty result, got %d tokens", len(tokens))
			}

			if !tt.wantEmpty && len(tokens) == 0 {
				t.Errorf("Expected non-empty result")
			}

			// Verify token offsets
			for i, token := range tokens {
				if token.StartByte < 0 || token.EndByte < token.StartByte {
					t.Errorf("Token %d has invalid offsets: start=%d, end=%d",
						i, token.StartByte, token.EndByte)
				}
			}
		})
	}
}

func TestTokenizerEncodeIDs(t *testing.T) {
	tok, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	input := "Hello, world!"
	ids, err := tok.EncodeIDs([]byte(input))
	if err != nil {
		t.Fatalf("EncodeIDs() error = %v", err)
	}

	tokens, err := tok.Encode([]byte(input))
	if err != nil {
		t.Fatalf("Encode() error = %v", err)
	}

	if len(ids) != len(tokens) {
		t.Errorf("EncodeIDs returned %d ids, Encode returned %d tokens", len(ids), len(tokens))
	}

	// Verify IDs match
	for i, id := range ids {
		if id != tokens[i].ID {
			t.Errorf("ID mismatch at %d: EncodeIDs=%d, Encode=%d", i, id, tokens[i].ID)
		}
	}
}

func TestTokenizerDecode(t *testing.T) {
	tok, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	input := "Hello, world!"
	ids, err := tok.EncodeIDs([]byte(input))
	if err != nil {
		t.Fatalf("EncodeIDs() error = %v", err)
	}

	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode() error = %v", err)
	}

	if string(decoded) != input {
		t.Errorf("Round-trip failed: got %q, want %q", string(decoded), input)
	}
}

func TestTokenizerTruncateToFit(t *testing.T) {
	tok, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	input := "The quick brown fox jumps over the lazy dog. " + strings.Repeat("More text here. ", 50)

	// First, count total tokens
	totalCount, err := tok.CountTokens([]byte(input))
	if err != nil {
		t.Fatalf("CountTokens() error = %v", err)
	}

	// Request truncation to half
	maxTokens := totalCount / 2
	cutoffByte, tokenCount, err := tok.TruncateToFit([]byte(input), maxTokens)
	if err != nil {
		t.Fatalf("TruncateToFit() error = %v", err)
	}

	if tokenCount != maxTokens {
		t.Errorf("TruncateToFit() returned tokenCount=%d, want %d", tokenCount, maxTokens)
	}

	if cutoffByte <= 0 || cutoffByte >= len(input) {
		t.Errorf("TruncateToFit() returned invalid cutoffByte=%d for input of len %d",
			cutoffByte, len(input))
	}

	// Verify truncated text tokenizes to expected count
	truncated := input[:cutoffByte]
	truncatedCount, err := tok.CountTokens([]byte(truncated))
	if err != nil {
		t.Fatalf("CountTokens() on truncated error = %v", err)
	}

	if truncatedCount > maxTokens {
		t.Errorf("Truncated text has %d tokens, expected <= %d", truncatedCount, maxTokens)
	}
}

func TestTokenizerStreamEncode(t *testing.T) {
	tok, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	input := "The quick brown fox jumps over the lazy dog. " + strings.Repeat("Additional sentence. ", 20)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream := tok.StreamEncode(ctx, []byte(input))

	// Collect results
	var results []ChunkResult
	for result := range stream.Chunks {
		if result.Err != nil {
			t.Fatalf("Stream error: %v", result.Err)
		}
		results = append(results, result)
	}

	// Wait for completion
	if err := stream.Wait(); err != nil {
		t.Fatalf("Stream.Wait() error = %v", err)
	}

	// Verify we got results
	if len(results) == 0 {
		t.Error("StreamEncode() returned no results")
	}

	// Verify total token count matches batch encode
	batchTokens, err := tok.Encode([]byte(input))
	if err != nil {
		t.Fatalf("Encode() error = %v", err)
	}

	totalStreamTokens := 0
	for _, r := range results {
		totalStreamTokens += len(r.Tokens)
	}

	if totalStreamTokens != len(batchTokens) {
		t.Errorf("StreamEncode got %d tokens, Encode got %d tokens",
			totalStreamTokens, len(batchTokens))
	}
}

func TestStreamCollector(t *testing.T) {
	collector := NewStreamCollector(3)

	// Add results out of order
	collector.Add(ChunkResult{Index: 2, Tokens: []Token{{ID: 3}}})
	collector.Add(ChunkResult{Index: 0, Tokens: []Token{{ID: 1}}})
	collector.Add(ChunkResult{Index: 1, Tokens: []Token{{ID: 2}}})

	if !collector.IsDone() {
		t.Error("Expected collector to be done")
	}

	tokens := collector.Tokens()
	if len(tokens) != 3 {
		t.Errorf("Expected 3 tokens, got %d", len(tokens))
	}

	// Verify order
	expectedIDs := []int{1, 2, 3}
	for i, token := range tokens {
		if token.ID != expectedIDs[i] {
			t.Errorf("Token %d has ID %d, want %d", i, token.ID, expectedIDs[i])
		}
	}
}

func TestTokenizerCancellation(t *testing.T) {
	tok, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	// Large input to ensure processing takes time
	input := strings.Repeat("This is a test sentence that should take some time to process. ", 1000)

	ctx, cancel := context.WithCancel(context.Background())

	stream := tok.StreamEncode(ctx, []byte(input))

	// Cancel immediately
	cancel()

	// Should complete quickly with error or partial results
	timeout := time.After(1 * time.Second)
	select {
	case <-stream.Done():
		// Expected
	case <-timeout:
		t.Error("StreamEncode did not respect cancellation within timeout")
	}
}

func TestCountTokens(t *testing.T) {
	tok, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	input := "Hello, world! This is a test."

	count, err := tok.CountTokens([]byte(input))
	if err != nil {
		t.Fatalf("CountTokens() error = %v", err)
	}

	tokens, err := tok.Encode([]byte(input))
	if err != nil {
		t.Fatalf("Encode() error = %v", err)
	}

	if count != len(tokens) {
		t.Errorf("CountTokens()=%d, len(Encode())=%d", count, len(tokens))
	}
}

// min is a helper for older Go versions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

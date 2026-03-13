package main

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	tokenizer "github.com/kraghavan/go-fast-token"
)

func main() {
	// Create tokenizer with custom config
	cfg := tokenizer.DefaultConfig()
	cfg.NumWorkers = 4
	cfg.MinChunkSize = 200 // Larger chunks for this example

	tok, err := tokenizer.New(cfg)
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %v", err)
	}

	// Simulate a large prompt
	prompt := generateLargePrompt()
	fmt.Printf("Input size: %d bytes\n", len(prompt))

	// Example 1: Streaming to simulated inference engine
	fmt.Println("\n=== Streaming Example ===")
	streamingExample(tok, prompt)

	// Example 2: Ordered streaming with collector
	fmt.Println("\n=== Ordered Collection Example ===")
	orderedExample(tok, prompt)

	// Example 3: Early truncation with cumulative counting
	fmt.Println("\n=== Cumulative Truncation Example ===")
	truncationExample(tok, prompt)
}

func streamingExample(tok tokenizer.Tokenizer, prompt []byte) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	stream := tok.StreamEncode(ctx, prompt)

	chunksReceived := 0
	totalTokens := 0
	start := time.Now()

	// Simulate prefill callback - in real use, this would send to inference engine
	for result := range stream.Chunks {
		if result.Err != nil {
			log.Printf("Chunk %d error: %v", result.Index, result.Err)
			continue
		}

		chunksReceived++
		totalTokens += len(result.Tokens)

		// Simulate inference engine accepting chunk
		fmt.Printf("  Chunk %d: %d tokens (received at %v)\n",
			result.Index, len(result.Tokens), time.Since(start).Round(time.Millisecond))

		// Simulate some prefill processing time
		time.Sleep(5 * time.Millisecond)
	}

	if err := stream.Wait(); err != nil {
		log.Printf("Stream error: %v", err)
	}

	fmt.Printf("\nTotal: %d chunks, %d tokens in %v\n",
		chunksReceived, totalTokens, time.Since(start).Round(time.Millisecond))
}

func orderedExample(tok tokenizer.Tokenizer, prompt []byte) {
	ctx := context.Background()
	stream := tok.StreamEncode(ctx, prompt)

	// Use ordered reader for in-order processing
	ordered := tokenizer.NewOrderedStreamReader(stream)

	var allTokens []tokenizer.Token
	for result := range ordered.Results() {
		if result.Err != nil {
			log.Printf("Error: %v", result.Err)
			continue
		}
		allTokens = append(allTokens, result.Tokens...)
	}

	<-ordered.Done()
	if err := ordered.Err(); err != nil {
		log.Printf("Ordered reader error: %v", err)
	}

	fmt.Printf("Collected %d tokens in order\n", len(allTokens))

	// Verify order by checking byte offsets are increasing
	lastEnd := 0
	outOfOrder := 0
	for _, t := range allTokens {
		if t.StartByte < lastEnd {
			outOfOrder++
		}
		lastEnd = t.EndByte
	}
	fmt.Printf("Out of order tokens: %d (should be 0)\n", outOfOrder)
}

func truncationExample(tok tokenizer.Tokenizer, prompt []byte) {
	ctx := context.Background()
	maxTokens := 500

	// Create truncation tracker
	truncator := tokenizer.NewCumulativeTruncation(maxTokens)

	stream := tok.StreamEncode(ctx, prompt)
	ordered := tokenizer.NewOrderedStreamReader(stream)

	chunksProcessed := 0
	for result := range ordered.Results() {
		if result.Err != nil {
			continue
		}

		chunksProcessed++
		foundCutoff, cutoffByte := truncator.ProcessChunk(result)

		if foundCutoff {
			fmt.Printf("Found cutoff at chunk %d, byte %d\n", chunksProcessed, cutoffByte)
			// In a real implementation, you could cancel the context here
			// to stop processing remaining chunks
			break
		}
	}

	// Get final result
	cutoffByte, tokenCount := truncator.Result()
	fmt.Printf("Truncation result: %d tokens, cutoff at byte %d\n", tokenCount, cutoffByte)
	fmt.Printf("Truncated text preview: %q...\n", string(prompt[:min(cutoffByte, 100)]))
}

func generateLargePrompt() []byte {
	base := `You are a helpful AI assistant. The user has provided the following context:

## Background
Large Language Models (LLMs) process text by first converting it into tokens.
This tokenization step is crucial for understanding how much context can fit
within the model's context window.

## Technical Details
Modern tokenizers like BPE (Byte Pair Encoding) learn to merge frequently
occurring byte sequences into single tokens. This allows efficient encoding
of common words while still handling rare words by breaking them into subwords.

## User Query
`

	// Add repetitive content to make it larger
	additional := strings.Repeat("Please analyze the following data point and provide insights. ", 100)

	return []byte(base + additional)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

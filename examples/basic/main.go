package main

import (
	"fmt"
	"log"

	tokenizer "github.com/kraghavan/go-fast-token"
)

func main() {
	// Create tokenizer with default config (GPT-4 / cl100k_base)
	tok, err := tokenizer.New(tokenizer.DefaultConfig())
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %v", err)
	}

	// Sample text
	text := `The transformer architecture has revolutionized natural language processing.
It uses self-attention mechanisms to process sequences in parallel, enabling
much larger models and faster training compared to recurrent neural networks.`

	// Encode with byte offsets
	tokens, err := tok.Encode([]byte(text))
	if err != nil {
		log.Fatalf("Encode failed: %v", err)
	}

	fmt.Printf("Input: %d bytes\n", len(text))
	fmt.Printf("Tokens: %d\n\n", len(tokens))

	// Show first 10 tokens with their source text
	fmt.Println("First 10 tokens:")
	for i, t := range tokens[:min(10, len(tokens))] {
		sourceText := text[t.StartByte:t.EndByte]
		fmt.Printf("  %d: ID=%d, bytes=[%d:%d], text=%q\n",
			i, t.ID, t.StartByte, t.EndByte, sourceText)
	}

	// Quick count without offsets
	count, _ := tok.CountTokens([]byte(text))
	fmt.Printf("\nQuick count: %d tokens\n", count)

	// Round-trip test
	ids, _ := tok.EncodeIDs([]byte(text))
	decoded, _ := tok.Decode(ids)
	fmt.Printf("\nRound-trip successful: %v\n", string(decoded) == text)

	// Truncation example
	maxTokens := 20
	cutoffByte, tokenCount, err := tok.TruncateToFit([]byte(text), maxTokens)
	if err != nil {
		log.Fatalf("TruncateToFit failed: %v", err)
	}

	fmt.Printf("\nTruncation to %d tokens:\n", maxTokens)
	fmt.Printf("  Cutoff byte: %d\n", cutoffByte)
	fmt.Printf("  Token count: %d\n", tokenCount)
	fmt.Printf("  Truncated text: %q...\n", text[:min(cutoffByte, 100)])
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

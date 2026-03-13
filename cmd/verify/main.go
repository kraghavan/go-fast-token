package main

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	tokenizer "github.com/kraghavan/go-fast-token"
	"github.com/pkoukk/tiktoken-go"
)

func main() {
	fmt.Println("=== go-fast-token Verification Suite ===\n")

	// Test texts of varying complexity
	testCases := []struct {
		name string
		text string
	}{
		{"simple", "Hello, world!"},
		{"sentence", "The quick brown fox jumps over the lazy dog."},
		{"multiline", "First line.\nSecond line.\nThird line."},
		{"unicode", "Hello 世界! Привет мир! 🚀"},
		{"code", "func main() { fmt.Println(\"Hello\") }"},
		{"long", strings.Repeat("This is a test sentence for tokenization. ", 100)},
	}

	tok, err := tokenizer.New(tokenizer.DefaultConfig())
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %v", err)
	}

	// Reference tiktoken encoder
	tiktok, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		log.Fatalf("Failed to create tiktoken: %v", err)
	}

	allPassed := true

	// Test 1: Correctness - match tiktoken exactly
	fmt.Println("## Test 1: Correctness (match tiktoken)")
	for _, tc := range testCases {
		// Our tokenizer
		ourIDs, _ := tok.EncodeIDs([]byte(tc.text))

		// Reference tiktoken
		refIDs := tiktok.Encode(tc.text, nil, nil)

		match := compareIDs(ourIDs, refIDs)
		status := "✅ PASS"
		if !match {
			status = "❌ FAIL"
			allPassed = false
		}
		fmt.Printf("  %s: %s (ours=%d, ref=%d tokens)\n", tc.name, status, len(ourIDs), len(refIDs))

		if !match {
			fmt.Printf("    Ours: %v\n", ourIDs[:min(10, len(ourIDs))])
			fmt.Printf("    Ref:  %v\n", refIDs[:min(10, len(refIDs))])
		}
	}

	// Test 2: Round-trip - tokenize → decode → original
	fmt.Println("\n## Test 2: Round-trip (encode → decode)")
	for _, tc := range testCases {
		ids, _ := tok.EncodeIDs([]byte(tc.text))
		decoded, _ := tok.Decode(ids)

		match := string(decoded) == tc.text
		status := "✅ PASS"
		if !match {
			status = "❌ FAIL"
			allPassed = false
		}
		fmt.Printf("  %s: %s\n", tc.name, status)

		if !match {
			fmt.Printf("    Original: %q\n", tc.text[:min(50, len(tc.text))])
			fmt.Printf("    Decoded:  %q\n", string(decoded)[:min(50, len(decoded))])
		}
	}

	// Test 3: Streaming ordering - stream result must match batch
	fmt.Println("\n## Test 3: Streaming ordering (stream == batch)")
	for _, tc := range testCases {
		// Batch encode
		batchTokens, _ := tok.Encode([]byte(tc.text))
		batchIDs := extractIDs(batchTokens)

		// Stream encode + reassemble
		ctx := context.Background()
		stream := tok.StreamEncode(ctx, []byte(tc.text))
		streamTokens, _ := tokenizer.CollectStream(ctx, stream)
		streamIDs := extractIDs(streamTokens)

		match := compareIDs(batchIDs, streamIDs)
		status := "✅ PASS"
		if !match {
			status = "❌ FAIL"
			allPassed = false
		}
		fmt.Printf("  %s: %s (batch=%d, stream=%d tokens)\n", tc.name, status, len(batchIDs), len(streamIDs))
	}

	// Test 4: Byte offset validity
	fmt.Println("\n## Test 4: Byte offset validity")
	for _, tc := range testCases {
		tokens, _ := tok.Encode([]byte(tc.text))
		valid := validateOffsets(tokens, []byte(tc.text))
		status := "✅ PASS"
		if !valid {
			status = "❌ FAIL"
			allPassed = false
		}
		fmt.Printf("  %s: %s\n", tc.name, status)
	}

	// Test 5: Truncation consistency
	fmt.Println("\n## Test 5: Truncation consistency")
	longText := strings.Repeat("Word ", 500)
	for _, maxTokens := range []int{10, 50, 100, 200} {
		cutoff, count, err := tok.TruncateToFit([]byte(longText), maxTokens)
		if err != nil {
			fmt.Printf("  max=%d: ❌ ERROR: %v\n", maxTokens, err)
			allPassed = false
			continue
		}

		// Verify truncated text tokenizes to expected count
		truncated := longText[:cutoff]
		actualCount, _ := tok.CountTokens([]byte(truncated))

		status := "✅ PASS"
		if actualCount > maxTokens {
			status = "❌ FAIL"
			allPassed = false
		}
		fmt.Printf("  max=%d: %s (reported=%d, actual=%d, cutoff=%d bytes)\n",
			maxTokens, status, count, actualCount, cutoff)
	}

	// Test 6: Real LLM API (optional - requires API key)
	fmt.Println("\n## Test 6: Real LLM API verification")
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" || apiKey == "asdasdaa" {
		fmt.Println("  ⏭️  SKIPPED (set ANTHROPIC_API_KEY to test)")
	} else {
		// We can't directly send tokens to Claude API (it expects text)
		// But we can verify our count matches what the API would use
		testText := "Explain quantum computing in one sentence."
		ourCount, _ := tok.CountTokens([]byte(testText))
		fmt.Printf("  Token count for API test: %d tokens\n", ourCount)
		fmt.Println("  ℹ️  Manual verification: send this text to Claude and compare usage")
	}

	// Summary
	fmt.Println("\n" + strings.Repeat("=", 50))
	if allPassed {
		fmt.Println("✅ ALL TESTS PASSED - Tokenizer is verified!")
	} else {
		fmt.Println("❌ SOME TESTS FAILED - Review output above")
		os.Exit(1)
	}
}

func compareIDs(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func extractIDs(tokens []tokenizer.Token) []int {
	ids := make([]int, len(tokens))
	for i, t := range tokens {
		ids[i] = t.ID
	}
	return ids
}

func validateOffsets(tokens []tokenizer.Token, input []byte) bool {
	for i, t := range tokens {
		if t.StartByte < 0 || t.EndByte > len(input) {
			fmt.Printf("    Token %d: invalid range [%d:%d] for input len %d\n",
				i, t.StartByte, t.EndByte, len(input))
			return false
		}
		if t.StartByte >= t.EndByte {
			fmt.Printf("    Token %d: start >= end [%d:%d]\n", i, t.StartByte, t.EndByte)
			return false
		}
	}
	return true
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

var _ = bytes.Equal // silence import

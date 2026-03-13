package bench

import (
	"context"
	"strings"
	"testing"

	tokenizer "github.com/kraghavan/go-fast-token"
)

// generateText creates test input of approximately the given size
func generateText(size int) []byte {
	sentence := "The quick brown fox jumps over the lazy dog. "
	repeat := (size / len(sentence)) + 1
	text := strings.Repeat(sentence, repeat)
	return []byte(text[:size])
}

func BenchmarkEncode_Small(b *testing.B) {
	tok, _ := tokenizer.New(tokenizer.DefaultConfig())
	input := generateText(100) // ~25 tokens

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(input)
	}
}

func BenchmarkEncode_Medium(b *testing.B) {
	tok, _ := tokenizer.New(tokenizer.DefaultConfig())
	input := generateText(1000) // ~250 tokens

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(input)
	}
}

func BenchmarkEncode_Large(b *testing.B) {
	tok, _ := tokenizer.New(tokenizer.DefaultConfig())
	input := generateText(10000) // ~2500 tokens

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(input)
	}
}

func BenchmarkEncode_VeryLarge(b *testing.B) {
	tok, _ := tokenizer.New(tokenizer.DefaultConfig())
	input := generateText(100000) // ~25000 tokens

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(input)
	}
}

func BenchmarkEncodeIDs_Large(b *testing.B) {
	tok, _ := tokenizer.New(tokenizer.DefaultConfig())
	input := generateText(10000)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = tok.EncodeIDs(input)
	}
}

func BenchmarkCountTokens_Large(b *testing.B) {
	tok, _ := tokenizer.New(tokenizer.DefaultConfig())
	input := generateText(10000)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = tok.CountTokens(input)
	}
}

// Compare parallel vs single worker
func BenchmarkEncode_SingleWorker(b *testing.B) {
	cfg := tokenizer.DefaultConfig()
	cfg.NumWorkers = 1
	tok, _ := tokenizer.New(cfg)
	input := generateText(10000)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(input)
	}
}

func BenchmarkEncode_4Workers(b *testing.B) {
	cfg := tokenizer.DefaultConfig()
	cfg.NumWorkers = 4
	tok, _ := tokenizer.New(cfg)
	input := generateText(10000)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(input)
	}
}

func BenchmarkEncode_8Workers(b *testing.B) {
	cfg := tokenizer.DefaultConfig()
	cfg.NumWorkers = 8
	tok, _ := tokenizer.New(cfg)
	input := generateText(10000)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(input)
	}
}

func BenchmarkStreamEncode_Large(b *testing.B) {
	tok, _ := tokenizer.New(tokenizer.DefaultConfig())
	input := generateText(10000)
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		stream := tok.StreamEncode(ctx, input)
		for range stream.Chunks {
			// Consume all chunks
		}
		_ = stream.Wait()
	}
}

func BenchmarkTruncateToFit(b *testing.B) {
	tok, _ := tokenizer.New(tokenizer.DefaultConfig())
	input := generateText(10000)
	maxTokens := 1000

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _, _ = tok.TruncateToFit(input, maxTokens)
	}
}

func BenchmarkSplitByBoundary(b *testing.B) {
	input := generateText(10000)
	minChunkSize := 100

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = tokenizer.SplitByBoundary(input, minChunkSize)
	}
}

// Memory efficiency test
func BenchmarkEncode_MemoryEfficiency(b *testing.B) {
	cfg := tokenizer.DefaultConfig()
	cfg.EnablePooling = true
	tok, _ := tokenizer.New(cfg)
	input := generateText(10000)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(input)
	}
}

// Throughput test - tokens per second
func BenchmarkThroughput(b *testing.B) {
	tok, _ := tokenizer.New(tokenizer.DefaultConfig())
	input := generateText(10000)

	var totalTokens int
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		tokens, _ := tok.Encode(input)
		totalTokens += len(tokens)
	}

	b.ReportMetric(float64(totalTokens)/b.Elapsed().Seconds(), "tokens/sec")
}
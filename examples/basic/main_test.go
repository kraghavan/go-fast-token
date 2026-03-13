package main

import (
	"testing"

	tokenizer "github.com/kraghavan/go-fast-token"
)

func TestBasicEncode(t *testing.T) {
	tok, err := tokenizer.New(tokenizer.DefaultConfig())
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	text := []byte("Hello, world! This is a test.")

	tokens, err := tok.Encode(text)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	if len(tokens) == 0 {
		t.Error("Expected tokens, got none")
	}

	// Verify byte offsets are valid
	for i, tok := range tokens {
		if tok.StartByte < 0 || tok.EndByte > len(text) || tok.StartByte >= tok.EndByte {
			t.Errorf("Token %d has invalid offsets: start=%d, end=%d", i, tok.StartByte, tok.EndByte)
		}
	}
}

func TestBasicRoundTrip(t *testing.T) {
	tok, err := tokenizer.New(tokenizer.DefaultConfig())
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	text := "Hello, world!"

	ids, err := tok.EncodeIDs([]byte(text))
	if err != nil {
		t.Fatalf("EncodeIDs failed: %v", err)
	}

	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	if string(decoded) != text {
		t.Errorf("Round-trip failed: got %q, want %q", string(decoded), text)
	}
}

func TestBasicTruncation(t *testing.T) {
	tok, err := tokenizer.New(tokenizer.DefaultConfig())
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	text := []byte("The quick brown fox jumps over the lazy dog. This is additional text to make it longer.")

	maxTokens := 10
	cutoffByte, tokenCount, err := tok.TruncateToFit(text, maxTokens)
	if err != nil {
		t.Fatalf("TruncateToFit failed: %v", err)
	}

	if tokenCount != maxTokens {
		t.Errorf("Expected %d tokens, got %d", maxTokens, tokenCount)
	}

	if cutoffByte <= 0 || cutoffByte > len(text) {
		t.Errorf("Invalid cutoff byte: %d", cutoffByte)
	}
}
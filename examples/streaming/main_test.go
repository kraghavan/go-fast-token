package main

import (
"context"
"strings"
"testing"
"time"

tokenizer "github.com/kraghavan/go-fast-token"
)

func TestStreamEncode(t *testing.T) {
tok, err := tokenizer.New(tokenizer.DefaultConfig())
if err != nil {
t.Fatalf("Failed to create tokenizer: %v", err)
}

input := []byte("The quick brown fox jumps over the lazy dog. " + strings.Repeat("More text. ", 20))

ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

stream := tok.StreamEncode(ctx, input)

var totalTokens int
var chunksReceived int

for chunk := range stream.Chunks {
if chunk.Err != nil {
t.Fatalf("Stream error: %v", chunk.Err)
}
chunksReceived++
totalTokens += len(chunk.Tokens)
}

if err := stream.Wait(); err != nil {
t.Fatalf("Stream.Wait() error: %v", err)
}

if chunksReceived == 0 {
t.Error("Expected chunks, got none")
}

if totalTokens == 0 {
t.Error("Expected tokens, got none")
}

batchTokens, _ := tok.Encode(input)
if totalTokens != len(batchTokens) {
t.Errorf("Stream got %d tokens, batch got %d", totalTokens, len(batchTokens))
}
}

func TestOrderedStreamReader(t *testing.T) {
tok, err := tokenizer.New(tokenizer.DefaultConfig())
if err != nil {
t.Fatalf("Failed to create tokenizer: %v", err)
}

input := []byte(strings.Repeat("Test sentence for streaming. ", 30))

ctx := context.Background()
stream := tok.StreamEncode(ctx, input)
ordered := tokenizer.NewOrderedStreamReader(stream)

var lastIndex int = -1
for chunk := range ordered.Results() {
if chunk.Err != nil {
t.Fatalf("Ordered stream error: %v", chunk.Err)
}
if chunk.Index <= lastIndex {
t.Errorf("Out of order: got index %d after %d", chunk.Index, lastIndex)
}
lastIndex = chunk.Index
}

<-ordered.Done()
if err := ordered.Err(); err != nil {
t.Fatalf("Ordered reader error: %v", err)
}
}

func TestStreamCollector(t *testing.T) {
collector := tokenizer.NewStreamCollector(3)

collector.Add(tokenizer.ChunkResult{Index: 2, Tokens: []tokenizer.Token{{ID: 30}}})
collector.Add(tokenizer.ChunkResult{Index: 0, Tokens: []tokenizer.Token{{ID: 10}}})
collector.Add(tokenizer.ChunkResult{Index: 1, Tokens: []tokenizer.Token{{ID: 20}}})

if !collector.IsDone() {
t.Error("Expected collector to be done")
}

tokens := collector.Tokens()
if len(tokens) != 3 {
t.Fatalf("Expected 3 tokens, got %d", len(tokens))
}

expectedIDs := []int{10, 20, 30}
for i, tok := range tokens {
if tok.ID != expectedIDs[i] {
t.Errorf("Token %d: got ID %d, want %d", i, tok.ID, expectedIDs[i])
}
}
}

func TestCumulativeTruncation(t *testing.T) {
tok, err := tokenizer.New(tokenizer.DefaultConfig())
if err != nil {
t.Fatalf("Failed to create tokenizer: %v", err)
}

input := []byte(strings.Repeat("Word ", 500))
maxTokens := 100

ctx := context.Background()
stream := tok.StreamEncode(ctx, input)
ordered := tokenizer.NewOrderedStreamReader(stream)

truncator := tokenizer.NewCumulativeTruncation(maxTokens)

var foundCutoff bool
var cutoffByte int

for chunk := range ordered.Results() {
if chunk.Err != nil {
continue
}
found, cutoff := truncator.ProcessChunk(chunk)
if found {
foundCutoff = true
cutoffByte = cutoff
break
}
}

if !foundCutoff {
t.Error("Expected to find cutoff")
}

if cutoffByte <= 0 {
t.Errorf("Invalid cutoff byte: %d", cutoffByte)
}

resultByte, resultCount := truncator.Result()
if resultCount != maxTokens {
t.Errorf("Expected %d tokens, got %d", maxTokens, resultCount)
}

if resultByte != cutoffByte {
t.Errorf("Result mismatch: got %d, expected %d", resultByte, cutoffByte)
}
}

func TestStreamCancellation(t *testing.T) {
tok, err := tokenizer.New(tokenizer.DefaultConfig())
if err != nil {
t.Fatalf("Failed to create tokenizer: %v", err)
}

input := []byte(strings.Repeat("Large input text. ", 1000))

ctx, cancel := context.WithCancel(context.Background())
stream := tok.StreamEncode(ctx, input)

cancel()

timeout := time.After(1 * time.Second)
select {
case <-stream.Done():
case <-timeout:
t.Error("Stream did not respect cancellation")
}
}

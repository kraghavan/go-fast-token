package tokenizer

import (
	"bytes"
	"unicode"
)

// boundaryChars defines safe split points (whitespace + common punctuation)
var boundaryChars = []byte(" \t\n\r.,;:!?")

// whitespaceChars for special handling
var whitespaceChars = []byte(" \t\n\r")

// SplitByBoundary divides input into chunks at natural boundaries.
// This avoids splitting mid-token which would corrupt BPE encoding.
//
// Algorithm:
//  1. Walk input looking for boundary characters
//  2. Accumulate bytes until minChunkSize is reached
//  3. Split at the next boundary after minChunkSize
//  4. Track byte offsets for each chunk
//
// IMPORTANT: Whitespace is kept with the FOLLOWING word, not the preceding one.
// This preserves BPE merges like " This" which are single tokens.
//
// Uses []byte throughout - no string conversions (Gemini tip #1).
func SplitByBoundary(input []byte, minChunkSize int) []Chunk {
	if len(input) == 0 {
		return nil
	}

	// Handle small inputs - no splitting needed
	if len(input) <= minChunkSize {
		return []Chunk{{
			Index:      0,
			Data:       input,
			ByteOffset: 0,
		}}
	}

	// Pre-allocate with estimate (Gemini tip #2)
	estimatedChunks := (len(input) / minChunkSize) + 1
	chunks := make([]Chunk, 0, estimatedChunks)

	pos := 0
	chunkIndex := 0

	for pos < len(input) {
		remaining := input[pos:]

		// If remaining is smaller than minChunkSize, take it all
		if len(remaining) <= minChunkSize {
			chunks = append(chunks, Chunk{
				Index:      chunkIndex,
				Data:       remaining,
				ByteOffset: pos,
			})
			break
		}

		// Find split point: first boundary after minChunkSize
		splitPoint := findBoundaryAfter(remaining, minChunkSize)

		chunks = append(chunks, Chunk{
			Index:      chunkIndex,
			Data:       remaining[:splitPoint],
			ByteOffset: pos,
		})

		pos += splitPoint
		chunkIndex++

		// NOTE: We do NOT skip leading whitespace anymore.
		// Whitespace should stay with the following word for proper BPE merging.
		// " This" is a single token — splitting it into " " + "This" breaks merges.
	}

	return chunks
}

// findBoundaryAfter finds the first boundary character at or after minPos.
// Returns the position to split AT (exclusive end of current chunk).
//
// For whitespace: split BEFORE the whitespace (don't include it)
// For punctuation: split AFTER the punctuation (include it)
//
// This ensures:
//   - "test." stays together in current chunk
//   - " This" stays together in next chunk (space + word = often single token)
func findBoundaryAfter(data []byte, minPos int) int {
	// Start searching from minPos
	searchStart := minPos
	if searchStart >= len(data) {
		return len(data)
	}

	// Look for boundary in remaining portion
	for i := searchStart; i < len(data); i++ {
		b := data[i]
		if isBoundary(b) {
			if isWhitespaceChar(b) {
				// Whitespace: split BEFORE it (don't include in current chunk)
				// This keeps " word" together in the next chunk
				return i
			} else {
				// Punctuation: split AFTER it (include in current chunk)
				// This keeps "word." together
				return i + 1
			}
		}
	}

	// No boundary found - take it all
	return len(data)
}

// SplitByWhitespace is a simpler variant that only splits on whitespace.
// Useful when you want punctuation to stay with words.
func SplitByWhitespace(input []byte, minChunkSize int) []Chunk {
	if len(input) == 0 {
		return nil
	}

	if len(input) <= minChunkSize {
		return []Chunk{{
			Index:      0,
			Data:       input,
			ByteOffset: 0,
		}}
	}

	// Use bytes.Fields for whitespace splitting, then recombine to hit minChunkSize
	fields := bytes.Fields(input)
	if len(fields) == 0 {
		return nil
	}

	chunks := make([]Chunk, 0, len(input)/minChunkSize+1)

	var currentChunk []byte
	currentOffset := 0
	chunkIndex := 0
	pos := 0

	for _, field := range fields {
		// Find actual position of this field in input
		fieldPos := bytes.Index(input[pos:], field)
		if fieldPos == -1 {
			continue
		}
		fieldPos += pos

		if currentChunk == nil {
			// Start new chunk - include any leading whitespace
			currentOffset = pos // Start from current pos, not field pos
			// Include whitespace before field
			currentChunk = append(currentChunk, input[pos:fieldPos]...)
		} else if fieldPos > pos {
			// Add whitespace between fields
			currentChunk = append(currentChunk, input[pos:fieldPos]...)
		}

		currentChunk = append(currentChunk, field...)
		pos = fieldPos + len(field)

		// Check if we've hit minimum size
		if len(currentChunk) >= minChunkSize {
			chunks = append(chunks, Chunk{
				Index:      chunkIndex,
				Data:       currentChunk,
				ByteOffset: currentOffset,
			})
			currentChunk = nil
			chunkIndex++
		}
	}

	// Don't forget the last chunk
	if len(currentChunk) > 0 {
		chunks = append(chunks, Chunk{
			Index:      chunkIndex,
			Data:       currentChunk,
			ByteOffset: currentOffset,
		})
	}

	return chunks
}

// SplitFixedWithBoundary splits into roughly equal chunks but respects boundaries.
// Useful when you want N workers to get roughly equal work.
func SplitFixedWithBoundary(input []byte, numChunks int) []Chunk {
	if len(input) == 0 || numChunks <= 0 {
		return nil
	}

	if numChunks == 1 {
		return []Chunk{{
			Index:      0,
			Data:       input,
			ByteOffset: 0,
		}}
	}

	targetSize := len(input) / numChunks
	if targetSize < 50 {
		targetSize = 50 // Minimum reasonable chunk
	}

	return SplitByBoundary(input, targetSize)
}

// Helper functions

func isBoundary(b byte) bool {
	return bytes.IndexByte(boundaryChars, b) >= 0
}

func isWhitespaceChar(b byte) bool {
	return bytes.IndexByte(whitespaceChars, b) >= 0
}

func isWhitespace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

// isWordBoundary checks if position is at a word boundary (Unicode-aware)
func isWordBoundary(data []byte, pos int) bool {
	if pos <= 0 || pos >= len(data) {
		return true
	}

	// Check if we're between a non-space and space (or vice versa)
	prevRune := rune(data[pos-1])
	currRune := rune(data[pos])

	prevIsSpace := unicode.IsSpace(prevRune)
	currIsSpace := unicode.IsSpace(currRune)

	return prevIsSpace != currIsSpace
}

// MergeSmallChunks combines chunks smaller than threshold with neighbors.
// Useful as post-processing step when initial split creates micro-chunks.
func MergeSmallChunks(chunks []Chunk, minSize int) []Chunk {
	if len(chunks) <= 1 {
		return chunks
	}

	merged := make([]Chunk, 0, len(chunks))
	var accumulator []byte
	accumulatorOffset := 0
	accumulatorIndex := 0

	for i, chunk := range chunks {
		if accumulator == nil {
			accumulator = make([]byte, 0, minSize*2)
			accumulatorOffset = chunk.ByteOffset
			accumulatorIndex = len(merged)
		}

		accumulator = append(accumulator, chunk.Data...)

		// Emit if we've hit size threshold or it's the last chunk
		if len(accumulator) >= minSize || i == len(chunks)-1 {
			merged = append(merged, Chunk{
				Index:      accumulatorIndex,
				Data:       accumulator,
				ByteOffset: accumulatorOffset,
			})
			accumulator = nil
		}
	}

	// Reindex
	for i := range merged {
		merged[i].Index = i
	}

	return merged
}

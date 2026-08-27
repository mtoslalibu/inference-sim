package format

import (
	"strings"
	"testing"
)

// payloads enumerates the input shapes the contracts have to survive: the empty
// string, whitespace, multibyte text, format verbs, payloads that already carry
// the delimiters, a NUL byte, and a large payload.
func payloads() []string {
	return []string{
		"",
		" ",
		"\t\n",
		"Hello there!",
		"Hello, Alice!",
		"Hello, Alice, Bob and Carol! (3 recipients)",
		"Zoë",
		"名前",
		"%s",             // a format verb in the payload
		"%d %v %!s(int)", // more of them, including a malformed-verb rendering
		"[",
		"]",
		"[already bracketed]",
		"]inverted[",
		"\x00",
		strings.Repeat("x", 10000),
	}
}

// TestFormat_BracketsWrapPayloadVerbatim covers BC-H2-1 (the result opens with
// "[", closes with "]", and is at least two bytes), BC-H2-2 (stripping exactly
// those two bytes returns the input unchanged — no trimming, escaping,
// re-casing, or truncation) and BC-H2-3 (the length delta is exactly two).
func TestFormat_BracketsWrapPayloadVerbatim(t *testing.T) {
	for _, in := range payloads() {
		got := Format(in)

		if len(got) < 2 {
			t.Errorf("Format(%q) = %q, want length at least 2 (BC-H2-1)", in, got)
			continue
		}
		if got[0] != '[' {
			t.Errorf("Format(%q) = %q, want it to begin with '[' (BC-H2-1)", in, got)
		}
		if got[len(got)-1] != ']' {
			t.Errorf("Format(%q) = %q, want it to end with ']' (BC-H2-1)", in, got)
		}
		if payload := got[1 : len(got)-1]; payload != in {
			t.Errorf("Format(%q) = %q; stripping the delimiters yields %q, want the input verbatim (BC-H2-2)", in, got, payload)
		}
		if len(got) != len(in)+2 {
			t.Errorf("Format(%q) has length %d, want %d (BC-H2-3)", in, len(got), len(in)+2)
		}
	}
}

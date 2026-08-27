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

// TestFormat_EmptyPayloadYieldsBarePair pins CLARIFICATION C2-3: the empty
// string is not special-cased or rejected, so it has a bracketed form like any
// other input. BC-H2-5's "never a bare pair" is a property of composing with a
// greeting, not of Format alone, and this test keeps the two from being
// conflated. The literal is derived from the contracts by hand (BC-H2-2 plus
// BC-H2-3 with an empty input admit exactly one answer), not captured from the
// implementation's output.
func TestFormat_EmptyPayloadYieldsBarePair(t *testing.T) {
	if got := Format(""); got != "[]" {
		t.Errorf("Format(%q) = %q, want %q (C2-3: the empty payload is not special-cased)", "", got, "[]")
	}
}

// maxNestingDepth is the deepest nesting the metamorphic law is exercised at.
// The law is depth-independent, so a small bound is enough to distinguish linear
// nesting from an idempotent implementation.
const maxNestingDepth = 5

// TestFormat_NestingIsLinearNotIdempotent covers BC-H2-4: applying Format n
// times nests exactly n delimiter pairs around an intact payload. Non-idempotence
// is asserted directly rather than left implied, so no caller can assume
// double-wrapping is a no-op.
func TestFormat_NestingIsLinearNotIdempotent(t *testing.T) {
	for _, in := range payloads() {
		for n := 1; n <= maxNestingDepth; n++ {
			got := in
			for i := 0; i < n; i++ {
				got = Format(got)
			}

			openRun, closeRun := strings.Repeat("[", n), strings.Repeat("]", n)
			if !strings.HasPrefix(got, openRun) || !strings.HasSuffix(got, closeRun) {
				t.Errorf("Format applied %d times to %q = %q, want prefix %q and suffix %q (BC-H2-4)", n, in, got, openRun, closeRun)
				continue
			}
			if len(got) != len(in)+2*n {
				t.Errorf("Format applied %d times to %q has length %d, want %d (BC-H2-4)", n, in, len(got), len(in)+2*n)
				continue
			}
			if payload := got[n : len(got)-n]; payload != in {
				t.Errorf("Format applied %d times to %q leaves payload %q, want %q intact (BC-H2-4)", n, in, payload, in)
			}

			// "Exactly n" is only well defined when the payload does not itself
			// start or end with a delimiter: nesting "[x]" once legitimately
			// yields a leading run of two. For those payloads the prefix, length,
			// and payload laws above already pin the count.
			if !strings.HasPrefix(in, "[") && !strings.HasSuffix(in, "]") {
				if gotOpen := len(got) - len(strings.TrimLeft(got, "[")); gotOpen != n {
					t.Errorf("Format applied %d times to %q has %d leading '[', want exactly %d (BC-H2-4)", n, in, gotOpen, n)
				}
				if gotClose := len(got) - len(strings.TrimRight(got, "]")); gotClose != n {
					t.Errorf("Format applied %d times to %q has %d trailing ']', want exactly %d (BC-H2-4)", n, in, gotClose, n)
				}
			}
		}

		// Non-idempotence, stated as its own assertion (BC-H2-4).
		if once, twice := Format(in), Format(Format(in)); once == twice {
			t.Errorf("Format(Format(%q)) == Format(%q) == %q; BC-H2-4 requires Format to nest, not to be idempotent", in, in, once)
		}
	}
}

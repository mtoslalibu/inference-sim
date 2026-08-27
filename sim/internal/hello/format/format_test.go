package format

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/internal/hello"
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
		got := bracket(in)

		if len(got) < 2 {
			t.Errorf("bracket(%q) = %q, want length at least 2 (BC-H2-1)", in, got)
			continue
		}
		if got[0] != '[' {
			t.Errorf("bracket(%q) = %q, want it to begin with '[' (BC-H2-1)", in, got)
		}
		if got[len(got)-1] != ']' {
			t.Errorf("bracket(%q) = %q, want it to end with ']' (BC-H2-1)", in, got)
		}
		if payload := got[1 : len(got)-1]; payload != in {
			t.Errorf("bracket(%q) = %q; stripping the delimiters yields %q, want the input verbatim (BC-H2-2)", in, got, payload)
		}
		if len(got) != len(in)+2 {
			t.Errorf("bracket(%q) has length %d, want %d (BC-H2-3)", in, len(got), len(in)+2)
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
	if got := bracket(""); got != "[]" {
		t.Errorf("bracket(%q) = %q, want %q (C2-3: the empty payload is not special-cased)", "", got, "[]")
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
				got = bracket(got)
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
		if once, twice := bracket(in), bracket(bracket(in)); once == twice {
			t.Errorf("bracket(bracket(%q)) == bracket(%q) == %q; BC-H2-4 requires Format to nest, not to be idempotent", in, in, once)
		}
	}
}

// recipientInputs enumerates the recipient-name shapes BC-H2-5 quantifies over,
// mirroring the input classes hello's own totality suite uses: no names, blank
// names, one name, many names, and blanks mixed with usable names.
func recipientInputs() [][]string {
	return [][]string{
		nil,
		{},
		{""},
		{" "},
		{"\t\n"},
		{"", ""},
		{"Alice"},
		{"  Alice  "},
		{"Alice", "Bob"},
		{"Alice", "Bob", "Carol"},
		{"", "Alice", " "},
		{"[", "]"},
		{"%s"},
		{"(2 recipients)"},
	}
}

// TestFormat_ComposedWithGreetIsNeverABarePair covers BC-H2-5: composing Format
// over a hello greeting never yields the bare "[]". This is BC-H1-1 (a greeting
// is never empty) surviving composition, so it is the one contract that spans
// both holes — and the reason this package's declared dependency on hello is
// exercised rather than asserted.
func TestFormat_ComposedWithGreetIsNeverABarePair(t *testing.T) {
	for _, in := range recipientInputs() {
		greeting := hello.Greet(in...)
		got := Format(in...)

		if len(got) < 3 {
			t.Errorf("Format(hello.Greet(%q)) = %q, want length at least 3 (BC-H2-5)", in, got)
		}
		if got == "[]" {
			t.Errorf("Format(hello.Greet(%q)) = %q, the bare delimiter pair; BC-H2-5 forbids it", in, got)
		}
		// Anchors BC-H2-5 to BC-H2-2/BC-H2-3 as well: were Format ever to trim or
		// truncate a greeting, the length law would catch it here too, not only in
		// the standalone payload sweep.
		if len(got) != len(greeting)+2 {
			t.Errorf("Format(hello.Greet(%q)) has length %d, want %d (BC-H2-3 under composition)", in, len(got), len(greeting)+2)
		}
		if !strings.Contains(got, greeting) {
			t.Errorf("Format(hello.Greet(%q)) = %q, want it to carry the greeting %q verbatim (BC-H2-2 under composition)", in, got, greeting)
		}
	}
}

// TestFormat_IsPureAndDeterministic covers BC-H2-6 (INV-6): repeated calls with
// the same input return identical strings within a process.
func TestFormat_IsPureAndDeterministic(t *testing.T) {
	for _, in := range payloads() {
		first := bracket(in)
		for i := 0; i < 100; i++ {
			if got := bracket(in); got != first {
				t.Fatalf("bracket(%q) returned %q on call %d but %q on call 1 (BC-H2-6)", in, got, i+2, first)
			}
		}
	}
}

// TestFormat_IsDeterministicAcrossProcesses covers BC-H2-6's cross-process leg.
// Re-running the test binary in a child process is the cheapest honest check; a
// value baked into the test would only restate the implementation. Same pattern
// as hello's TestGreet_IsDeterministicAcrossProcesses.
func TestFormat_IsDeterministicAcrossProcesses(t *testing.T) {
	const key = "HELLO_FORMAT_SUBPROCESS"
	const recipient = "Alice"

	if os.Getenv(key) == "1" {
		fmt.Print(Format(recipient))
		return
	}

	exe, err := os.Executable()
	if err != nil {
		t.Skipf("cannot locate test binary: %v", err)
	}
	cmd := exec.Command(exe, "-test.run", "^"+t.Name()+"$")
	cmd.Env = append(os.Environ(), key+"=1")
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("subprocess failed: %v", err)
	}
	if want := Format(recipient); !strings.Contains(string(out), want) {
		t.Errorf("subprocess output %q does not contain the in-process result %q (BC-H2-6)", string(out), want)
	}
}

// FuzzBracket quantifies BC-H2-1, BC-H2-2 and BC-H2-3 over arbitrary bytes
// rather than over the payloads() table. The table is a fixed 16 entries, so a
// bracket that silently dropped some byte class outside it — a stray "\r", say —
// would satisfy every table-driven assertion while violating all three
// contracts. The laws are cheap to state universally, so state them universally.
func FuzzBracket(f *testing.F) {
	for _, seed := range payloads() {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, payload string) {
		got := bracket(payload)
		if !strings.HasPrefix(got, "[") || !strings.HasSuffix(got, "]") || len(got) < 2 {
			t.Fatalf("bracket(%q) = %q, want it delimited by '[' and ']' (BC-H2-1)", payload, got)
		}
		if inner := got[1 : len(got)-1]; inner != payload {
			t.Fatalf("bracket(%q) = %q; interior is %q, want the payload verbatim (BC-H2-2)", payload, got, inner)
		}
		if len(got) != len(payload)+2 {
			t.Fatalf("bracket(%q) has length %d, want %d (BC-H2-3)", payload, len(got), len(payload)+2)
		}
	})
}

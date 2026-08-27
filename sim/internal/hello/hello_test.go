package hello

import (
	"fmt"
	"os"
	"os/exec"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// blankInputs enumerates every shape of "no usable recipient" the contracts name.
func blankInputs() [][]string {
	return [][]string{
		nil,
		{},
		{""},
		{" "},
		{"\t"},
		{"\n"},
		{"", ""},
		{" ", "\t\n", ""},
	}
}

// TestGreet_BlankInputYieldsSameNonEmptyGreeting covers BC-H1-1 (the returned
// string is never empty) and the first half of BC-H1-3 (every blank input maps
// to one and the same generic greeting).
func TestGreet_BlankInputYieldsSameNonEmptyGreeting(t *testing.T) {
	want := Greet()
	if len(want) == 0 {
		t.Fatalf("Greet() returned the empty string; BC-H1-1 requires length > 0")
	}
	for _, in := range blankInputs() {
		if got := Greet(in...); got != want {
			t.Errorf("Greet(%q) = %q, want %q (BC-H1-3: all blank input maps to one generic greeting)", in, got, want)
		}
	}
}

// TestGreet_SingleNameIsCarriedThrough covers BC-H1-2 (a single non-blank name
// appears in the greeting in its trimmed form) and the second half of BC-H1-3
// (non-blank input never yields the generic greeting).
func TestGreet_SingleNameIsCarriedThrough(t *testing.T) {
	cases := []struct{ in, wantSub string }{
		{"Alice", "Alice"},
		{"  Alice  ", "Alice"},
		{"\tBob\n", "Bob"},
		{"Ada Lovelace", "Ada Lovelace"},
		{"Ada  Lovelace", "Ada  Lovelace"}, // interior whitespace is not collapsed
		{"Zo\u00eb", "Zo\u00eb"},
		{"\u540d\u524d", "\u540d\u524d"},
		{"O'Brien", "O'Brien"},
	}
	for _, c := range cases {
		got := Greet(c.in)
		if !strings.Contains(got, c.wantSub) {
			t.Errorf("Greet(%q) = %q, want it to contain %q (BC-H1-2)", c.in, got, c.wantSub)
		}
		if got == Greet() {
			t.Errorf("Greet(%q) = %q, which equals the generic greeting; BC-H1-3 forbids that for non-blank input", c.in, got)
		}
		if len(got) == 0 {
			t.Errorf("Greet(%q) returned the empty string (BC-H1-1)", c.in)
		}
	}
}

// TestGreet_BlankNamesAreIgnoredAmongUsableOnes covers BC-H1-2/BC-H1-3: blank
// names mixed in with usable ones are ignored, not carried through.
func TestGreet_BlankNamesAreIgnoredAmongUsableOnes(t *testing.T) {
	got := Greet("", "  ", "Alice", "\t")
	want := Greet("Alice")
	if got != want {
		t.Errorf("Greet with padding blanks = %q, want %q (blank names are ignored)", got, want)
	}
}

// TestGreet_MultipleNamesAreAllCarriedThrough covers BC-H1-2 extended to the
// multi-recipient path: every usable name appears in the greeting.
func TestGreet_MultipleNamesAreAllCarriedThrough(t *testing.T) {
	names := []string{" Alice ", "Bob", "Carol"}
	got := Greet(names...)
	for _, n := range names {
		if want := strings.TrimSpace(n); !strings.Contains(got, want) {
			t.Errorf("Greet(%q) = %q, want it to contain %q (BC-H1-2)", names, got, want)
		}
	}
	if got == Greet() {
		t.Errorf("Greet(%q) = %q, which equals the generic greeting (BC-H1-3)", names, got)
	}
}

// countRE extracts the recipient count from the tail of a multi-recipient
// greeting. It is anchored at the end of the string so a recipient name that
// happens to look like the suffix cannot be mistaken for it.
var countRE = regexp.MustCompile(`\((\d+) recipients\)$`)

// statedCount returns the count a multi-recipient greeting states.
func statedCount(t *testing.T, greeting string) int {
	t.Helper()
	m := countRE.FindStringSubmatch(greeting)
	if m == nil {
		t.Fatalf("greeting %q states no recipient count (BC-H1-4)", greeting)
	}
	n, err := strconv.Atoi(m[1])
	if err != nil {
		t.Fatalf("greeting %q has an unparseable count %q: %v", greeting, m[1], err)
	}
	return n
}

// TestGreet_StatedCountEqualsNamesCarriedThrough covers BC-H1-4: the count the
// greeting states equals the number of names it carries through per BC-H1-2.
func TestGreet_StatedCountEqualsNamesCarriedThrough(t *testing.T) {
	names := []string{"Alice", "Bob", "Carol", "Dave"}
	for n := 2; n <= len(names); n++ {
		in := names[:n]
		got := Greet(in...)
		if c := statedCount(t, got); c != n {
			t.Errorf("Greet(%q) states %d recipients, want %d (BC-H1-4)", in, c, n)
		}
		for _, name := range in {
			if !strings.Contains(got, name) {
				t.Errorf("Greet(%q) = %q, want it to contain %q (BC-H1-2 under BC-H1-4)", in, got, name)
			}
		}
	}
}

// TestGreet_CountMetamorphicUnderAddingNames covers BC-H1-4's metamorphic law:
// one more distinct non-blank name raises the stated count by exactly one, and
// a blank name does not change it.
func TestGreet_CountMetamorphicUnderAddingNames(t *testing.T) {
	extra := []string{"Carol", "Dave", "Erin", "Frank"}
	in := []string{"Alice", "Bob"}
	prev := statedCount(t, Greet(in...))
	if prev != 2 {
		t.Fatalf("baseline count = %d, want 2", prev)
	}
	for _, name := range extra {
		in = append(in, name)
		got := statedCount(t, Greet(in...))
		if got != prev+1 {
			t.Errorf("after adding %q the stated count = %d, want %d (BC-H1-4)", name, got, prev+1)
		}
		prev = got

		for _, blank := range []string{"", " ", "\t\n"} {
			withBlank := append(append([]string{}, in...), blank)
			if c := statedCount(t, Greet(withBlank...)); c != prev {
				t.Errorf("adding blank %q changed the stated count to %d, want %d (BC-H1-4)", blank, c, prev)
			}
		}
	}
}

// TestGreet_DuplicateNamesAreCounted pins CLARIFICATION C-2: names are counted,
// not de-duplicated, which is what makes BC-H1-4's law hold unconditionally.
func TestGreet_DuplicateNamesAreCounted(t *testing.T) {
	if c := statedCount(t, Greet("Alice", "Alice", "Alice")); c != 3 {
		t.Errorf("Greet with three identical names states %d recipients, want 3 (C-2: no de-duplication)", c)
	}
}

// TestGreet_SingleNameStatesNoCount pins CLARIFICATION C-3: a one-recipient
// greeting states no count, so no "(1 recipients)" is ever produced.
func TestGreet_SingleNameStatesNoCount(t *testing.T) {
	if got := Greet("Alice"); countRE.MatchString(got) {
		t.Errorf("Greet(\"Alice\") = %q, want no stated count (C-3)", got)
	}
}

// TestGreet_TotalityOverAdversarialInput covers BC-H1-1 over adversarial but
// ordinary input. Library code must not panic on ordinary input, so this also
// guards the error-handling boundary in principles.md.
func TestGreet_TotalityOverAdversarialInput(t *testing.T) {
	inputs := [][]string{
		nil,
		{},
		{""},
		{" "},                                // non-breaking space: unicode.IsSpace treats it as blank
		{"%s"},                               // format verb in a name
		{"%d", "%v"},                         // format verbs, multi path
		{"(2 recipients)"},                   // name that mimics the count suffix
		{"(2 recipients)", "(9 recipients)"}, // ...on the multi path
		{"", "Alice", ""},
		{strings.Repeat("x", 10000)},
		{strings.Repeat("x", 1000), strings.Repeat("y", 1000)},
		{"Alice", "", "Bob", " ", "Carol"},
		{"[", "]"},
		{"\x00"},
	}
	for _, in := range inputs {
		got := Greet(in...)
		if len(got) == 0 {
			t.Errorf("Greet(%q) returned the empty string; BC-H1-1 requires length > 0", in)
		}
	}
}

// TestGreet_NameMimickingCountSuffixDoesNotConfuseTheCount covers BC-H1-4: a
// recipient name that looks like the count suffix must not be read as the count.
func TestGreet_NameMimickingCountSuffixDoesNotConfuseTheCount(t *testing.T) {
	if c := statedCount(t, Greet("(9 recipients)", "Bob")); c != 2 {
		t.Errorf("stated count = %d, want 2 (a name that mimics the suffix must not be read as the count)", c)
	}
}

// TestGreet_IsPureAndDeterministic covers BC-H1-5 (INV-6): repeated calls with
// the same input return identical strings.
func TestGreet_IsPureAndDeterministic(t *testing.T) {
	inputs := append(blankInputs(),
		[]string{"Alice"},
		[]string{"Alice", "Bob"},
		[]string{"Alice", "Bob", "Carol"},
		[]string{" Alice ", "", "Bob"},
	)
	for _, in := range inputs {
		first := Greet(in...)
		for i := 0; i < 100; i++ {
			if got := Greet(in...); got != first {
				t.Fatalf("Greet(%q) returned %q on call %d but %q on call 1 (BC-H1-5)", in, got, i+2, first)
			}
		}
	}
}

// TestGreet_DoesNotMutateItsArgument covers BC-H1-5: a caller reusing its slice
// must see it unchanged, and a second call must agree with the first.
func TestGreet_DoesNotMutateItsArgument(t *testing.T) {
	in := []string{"  Alice  ", "", "Bob"}
	before := append([]string{}, in...)
	first := Greet(in...)
	if !reflect.DeepEqual(in, before) {
		t.Errorf("Greet mutated its argument: %q, want %q (BC-H1-5)", in, before)
	}
	if second := Greet(in...); second != first {
		t.Errorf("second call returned %q, want %q (BC-H1-5)", second, first)
	}
}

// TestGreet_IsDeterministicAcrossProcesses covers BC-H1-5's cross-process leg.
// Re-running the test binary in a child process is the cheapest honest check;
// a value baked into the test would only restate the implementation.
func TestGreet_IsDeterministicAcrossProcesses(t *testing.T) {
	const key = "HELLO_GREET_SUBPROCESS"
	args := []string{"Alice", "Bob", "Carol"}
	if os.Getenv(key) == "1" {
		fmt.Print(Greet(args...))
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
	want := Greet(args...)
	if got := string(out); !strings.Contains(got, want) {
		t.Errorf("subprocess output %q does not contain the in-process greeting %q (BC-H1-5)", got, want)
	}
}

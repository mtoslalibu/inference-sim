package hello

import (
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

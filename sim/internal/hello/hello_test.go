package hello

import "testing"

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

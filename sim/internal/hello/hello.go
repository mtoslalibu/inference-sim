// Package hello produces human-readable greetings for zero or more recipient
// names. Its single guarantee is totality: every input, including no input at
// all, maps to a non-empty greeting string. There is no failure mode and hence
// no error return.
package hello

import (
	"fmt"
	"strings"

	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// genericGreeting is returned when no usable recipient name is supplied. It is
// the only value Greet returns for blank input, and Greet never returns it for
// input that carries at least one name (BC-H1-3).
const genericGreeting = "Hello there!"

// singleFormat greets a recipient list. The comma after "Hello" is what keeps
// every non-blank result distinct from genericGreeting (BC-H1-3).
const singleFormat = "Hello, %s!"

// multiFormat greets two or more recipients and states how many it names. The
// count is the trailing element so it can be read unambiguously off the end of
// the greeting, even when a recipient name mimics this suffix.
const multiFormat = "Hello, %s! (%d recipients)"

// Greet returns a greeting for the given recipient names. Names that are empty
// or whitespace-only are ignored. With no usable name it returns a generic
// greeting. Greet is pure: it reads no clock, no RNG, no environment, and no
// package-level mutable state, and it iterates no map (BC-H1-5, INV-6, R2).
func Greet(names ...string) string {
	kept := usableNames(names)
	switch len(kept) {
	case 0:
		return genericGreeting
	case 1:
		return fmt.Sprintf(singleFormat, kept[0])
	default:
		return fmt.Sprintf(multiFormat, joinNames(kept), util.Len64(kept))
	}
}

// usableNames returns the trimmed, non-blank names in their original order.
// Order is preserved from the argument slice, so the result is deterministic and
// no map is involved (BC-H1-5, R2). The argument slice is never written to.
func usableNames(names []string) []string {
	kept := make([]string, 0, len(names))
	for _, n := range names {
		if trimmed := strings.TrimSpace(n); trimmed != "" {
			kept = append(kept, trimmed)
		}
	}
	return kept
}

// joinNames renders the kept names as a human-readable list: "A" for one,
// "A and B" for two, "A, B and C" for three or more. Order follows the argument
// order. Callers must pass at least one name.
func joinNames(kept []string) string {
	last := len(kept) - 1
	if last == 0 {
		return kept[0]
	}
	return strings.Join(kept[:last], ", ") + " and " + kept[last]
}

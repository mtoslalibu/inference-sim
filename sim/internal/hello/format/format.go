// Package format presents a greeting for display by delimiting it with
// brackets. It obtains the greeting from sim/internal/hello and adds exactly one
// presentation concern on top of it without altering it: the delimiter style is
// a contract-tested guarantee owned here rather than an informal habit repeated
// at each call site.
//
// Every greeting, including the one produced for no recipients, has a valid
// bracketed form, so there is no failure mode and hence no error return. The
// greeting is carried through byte-for-byte — nothing is trimmed, escaped,
// re-cased, or truncated.
//
// The brackets are hard-coded: there is no delimiter parameter and no delimiter
// strategy interface. One consumer shape, one delimiter; a seam for a
// hypothetical second style would freeze a surface no caller exercises. A second
// style later is a one-line edit plus one contract set.
//
// This package must reach utility behavior through sim/internal/hello, never
// around it, so the two dependency arrows stay a chain rather than a triangle:
// sim/internal/util is deliberately not imported here. The hello import is
// one-directional by construction — if hello ever imported this package the
// build would break with an import cycle.
package format

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim/internal/hello"
)

// delimitedFormat is the sole delimiter style. The payload is passed as an
// argument rather than interpolated into the format string, so a payload
// containing format verbs is emitted verbatim (BC-H2-2).
const delimitedFormat = "[%s]"

// Format returns the greeting for the given recipient names, wrapped in a
// leading '[' and a trailing ']'. The greeting itself comes from hello.Greet, so
// this package owns presentation only and never restates greeting behavior.
//
// The result always begins with '[' and ends with ']' (BC-H2-1), its interior is
// the greeting byte-for-byte (BC-H2-2), its length is exactly the greeting's plus
// two (BC-H2-3), and it is never the bare delimiter pair, because a greeting is
// never empty (BC-H2-5, inherited from BC-H1-1).
//
// Format is pure: it reads no clock, no RNG, no environment, and no
// package-level mutable state, and it iterates no map (BC-H2-6, INV-6, R2).
func Format(names ...string) string {
	return bracket(hello.Greet(names...))
}

// bracket is the delimiter law in isolation, separated from Format so the
// payload-level contracts (BC-H2-1 through BC-H2-4) can be quantified over
// arbitrary strings rather than only over strings hello.Greet happens to
// produce. Applying it n times nests n delimiter pairs — it is deliberately not
// idempotent (BC-H2-4).
func bracket(payload string) string {
	return fmt.Sprintf(delimitedFormat, payload)
}

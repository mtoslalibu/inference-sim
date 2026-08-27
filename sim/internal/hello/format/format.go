// Package format delimits an already-produced greeting with brackets for
// presentation. It adds exactly one presentation concern on top of the greeting
// produced by sim/internal/hello without altering it: the delimiter style is a
// contract-tested guarantee owned here rather than an informal habit repeated at
// each call site.
//
// Every input string, including the empty string, has a valid bracketed form, so
// there is no failure mode and hence no error return. The payload is carried
// through byte-for-byte — nothing is trimmed, escaped, re-cased, or truncated.
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

import "fmt"

// delimitedFormat is the sole delimiter style. The payload is passed as an
// argument rather than interpolated into the format string, so a payload
// containing format verbs is emitted verbatim (BC-H2-2).
const delimitedFormat = "[%s]"

// Format returns the given greeting wrapped in a leading '[' and a trailing ']'.
// The result always begins with '[' and ends with ']' (BC-H2-1), its interior is
// the input byte-for-byte (BC-H2-2), and its length is exactly the input's plus
// two (BC-H2-3). Applying Format n times nests n delimiter pairs — it is
// deliberately not idempotent (BC-H2-4).
//
// Format is pure: it reads no clock, no RNG, no environment, and no
// package-level mutable state, and it iterates no map (BC-H2-6, INV-6, R2).
func Format(greeting string) string {
	return fmt.Sprintf(delimitedFormat, greeting)
}

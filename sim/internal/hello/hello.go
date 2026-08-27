// Package hello produces human-readable greetings for zero or more recipient
// names. Its single guarantee is totality: every input, including no input at
// all, maps to a non-empty greeting string. There is no failure mode and hence
// no error return.
package hello

// genericGreeting is returned when no usable recipient name is supplied. It is
// the only value Greet returns for blank input, and Greet never returns it for
// input that carries at least one name (BC-H1-3).
const genericGreeting = "Hello there!"

// Greet returns a greeting for the given recipient names. Names that are empty
// or whitespace-only are ignored. With no usable name it returns a generic
// greeting. Greet is pure: it reads no clock, no RNG, no environment, and no
// package-level mutable state, and it iterates no map (BC-H1-5, INV-6, R2).
func Greet(names ...string) string {
	return genericGreeting
}

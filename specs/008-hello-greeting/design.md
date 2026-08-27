<!-- RFC prose for tracking issue #6: RFC: hello greeting utility (sim/internal/hello + sim/internal/hello/format) -->
<!-- Source: https://github.com/mtoslalibu/inference-sim/issues/6 — copied verbatim at .archon encoding time. -->

# RFC: hello greeting utility (`sim/internal/hello`, `sim/internal/hello/format`)

Filed per `docs/contributing/rfc.md`. An RFC applies because this introduces **two new package boundaries** (not a policy behind an existing interface, not a single-file refactor).

---

## Step 0: Baseline Analysis

Ran the [archon baseline analysis](docs/contributing/archon/baseline-analysis.md) with archon-go `v0.3.0` (pinned in `.archon-version`, built via `scripts/archon-build.sh`).

**`health`**

```
cycles: none — internal dependency graph is an acyclic DAG (healthy)
god-modules (high fan-in + large surface): sim, workload
coupling (top by blast radius):
  package     fanIn  fanOut  surf  instab  blast
  tokenid         3       0     1    0.00     11
  hash            3       1     2    0.25     10
  util            2       0     1    0.00      9
  sim             7       3   315    0.30      8  <god>
  kvkey           1       2     8    0.67      4
```

**`impact sim/internal/util`** (the dependency this RFC declares)

```
2 direct dependent(s), 9 total (transitive)
direct:   sim, kv
indirect: inference-sim, cmd, cluster, latency, lora, saturation, workload
```

**`evidence`** — 498 lines of contract/evidence output; every listed contract test reports `CI: PASS`. Most implementer attributions read `unconfirmed (a contract test exists but drives implementers via a factory, so this one cannot be attributed)` — a pre-existing, repo-wide attribution limitation, not a regression introduced here.

**How this informs the design:**

- **Safe to add here.** The graph is an acyclic DAG today, and the leaf-utility corner (`util`, `hash`, `tokenid`: fanOut 0–1, instability 0.00–0.25, surface 1–2 symbols) is the healthiest region of the repo. Two new leaf packages with a 1-symbol surface each fit that established shape.
- **`util` has blast radius 9 (transitive), fanOut 0, instability 0.00.** We are *consuming* it, not changing it, so the blast radius is not spent — but it does mean `util` must stay a sink. This RFC adds no export to `util` and no import *into* it.
- **Neither god-module is touched.** `sim` (surface 315, fanIn 7) and `workload` (surface 158) are untouched: no new symbol enters either, so their surfaces do not grow.
- **The cycle risk is the one real architectural hazard,** and it is specific to the requested shape (a child package importing its parent). See Trade-offs.

---

## Section 1: Motivation & Scope

**What problem does this solve?**

There is no shared, contract-tested way to produce a human-readable greeting string inside `sim/`. Where greeting-like text is needed (diagnostics, banners, operator-facing messages), each call site formats its own string ad hoc, so nothing guarantees the result is non-empty and nothing guarantees a consistent delimiter style. A caller today cannot depend on "this string is safe to print" without re-checking it.

**What:** two leaf packages under `sim/internal/`. `hello` produces a greeting for zero or more recipient names, with a total guarantee that the result is never the empty string. `hello/format` is a thin presentation wrapper that delimits an existing greeting with brackets.

**Why:** it turns two properties that are currently informal (non-emptiness, delimiter style) into contract-tested guarantees owned by one package each, so callers stop defensively re-checking and stop drifting in style.

**How it behaves:** no user-visible behavior change. **No CLI flag, no config key, no trace field, no stdout change.** These are internal library packages; nothing in `cmd/` or the DES is wired to them in this RFC. A user running `blis` sees byte-identical output before and after (INV-6).

**Scope in**

- `hello` exports one greeting function taking zero or more recipient names.
- `hello/format` exports one bracket-delimiting function.
- Contracts + evidence for both (property and metamorphic tests).

**Scope out**

- Any call-site adoption or replacement of existing ad-hoc greeting strings — deliberately deferred so the packages land with zero blast radius, and adoption can be reviewed per call site.
- CLI surface, config surface, trace/metric fields.
- Localization / i18n, pluralization rules, non-ASCII casing rules.
- Alternative delimiter styles (parentheses, quotes) and a delimiter-strategy interface — see Trade-offs.
- Any wiring into `sim`, `workload`, or `cmd`.

**Modeling decisions**

This feature models no physical or serving-system behavior, so the usual modeled/simplified/omitted table has no fidelity axis. Two presentation decisions are recorded because they are the ones a reviewer could reasonably contest:

| Aspect | Decision | Justification / what is lost |
|---|---|---|
| Recipient count in the greeting | Modeled — a multi-recipient greeting reports how many recipients it names | Makes the declared `util` dependency load-bearing rather than decorative (see Trade-offs). Lost: nothing that exists today. |
| Empty / whitespace-only recipient input | Simplified to a single generic fallback greeting, not per-case diagnostics | Preserves the never-empty total guarantee with one code path. Lost: the caller cannot distinguish "no names given" from "names given but all blank" from the returned string alone. Accepted because the packages are presentation-only; a caller needing that distinction should validate before calling. |
| Localization | Omitted | No consumer needs it; adding a locale parameter now would freeze a surface no caller exercises. Lost: non-English greetings. |

---

## Section 2: Holes (architectural intent)

### H1: `sim/internal/hello`

| Field | Value |
|---|---|
| **Name** | `sim/internal/hello` |
| **Responsibility** | Produces a human-readable greeting for zero or more recipient names, guaranteeing a non-empty result. |
| **Surface** | One exported function, `Greet`: takes zero or more recipient names and returns a single greeting string. Given one name it greets that name; given several it greets them and states how many recipients it names; given none, or only names that are empty/whitespace-only after trimming, it returns a generic greeting. It returns only a string — no error — because it has no failure mode: every input maps to a valid greeting. |
| **Allowed imports** | `sim/internal/util` (the requested dependency); Go stdlib `strings`, `fmt`. **Denied:** everything else in the repo — in particular `sim`, `sim/workload`, `sim/kv`, `sim/latency`, any `cmd/` package, and `sim/internal/hello/format` (importing the child would create the one cycle this design must avoid). |
| **Contracts** | **BC-H1-1 (totality / never empty):** GIVEN any recipient-name input whatsoever — none, one, many, empty strings, whitespace-only strings, or any mix — WHEN `Greet` is called, THEN the returned string has length greater than zero. *[evidenced: property_test]*<br>**BC-H1-2 (name is carried through):** GIVEN a single recipient name that is non-empty after trimming, WHEN `Greet` is called, THEN the returned greeting contains that trimmed name as a substring. *[evidenced: property_test]*<br>**BC-H1-3 (blank input degrades to the generic greeting, and only then):** GIVEN input consisting only of empty or whitespace-only names (or no names), WHEN `Greet` is called, THEN the result equals the same generic greeting for every such input — and GIVEN at least one name that is non-empty after trimming, the result does NOT equal that generic greeting. *[evidenced: property_test]*<br>**BC-H1-4 (count agreement):** GIVEN two or more recipient names that are non-empty after trimming, WHEN `Greet` is called, THEN the count stated in the greeting equals the number of names it actually carries through per BC-H1-2. *[evidenced: metamorphic_test — adding one more distinct non-blank name increases the stated count by exactly one; adding a blank name does not change it]*<br>**BC-H1-5 (purity / determinism):** GIVEN identical recipient-name input, WHEN `Greet` is called repeatedly within a process and across processes, THEN it returns identical strings. It reads no clock, no RNG, no environment, and no package-level mutable state, and it iterates no map. *[evidenced: property_test]* |
| **Evidence type** | property_test (BC-H1-1, -2, -3, -5), metamorphic_test (BC-H1-4). `differential_test` is deliberately **not** claimed for any contract: there is no reference implementation or external oracle to differentiate against, and asserting one would be theatre. |
| **Invariants** | **INV-6 (determinism)** is the only one of the 13 that this hole can affect, and BC-H1-5 is how it is preserved: pure, no RNG/clock, no map iteration (R2). It is preserved *vacuously* at the run level as well, since nothing calls `Greet` on any simulator path in this RFC. INV-1/2/4/5/8/12 (conservation, lifecycle, KV, causality, work-conserving, batch completeness) are **not applicable** — this hole touches no request, no KV block, no event, and no clock. INV-13 (run/replay parity) is not applicable: nothing is written to or read from a trace. Stating this explicitly rather than padding the list with invariants the hole cannot influence. |
| **Extension type** | **subsystem module** — a new package with its own surface, not a variant behind an existing interface. Recorded honestly: the four-way taxonomy in the design guidelines is aimed at simulator subsystems, and a leaf string utility strains it. `subsystem module` is the closest fit because the package is new and self-owned; it is emphatically *not* a policy template (no interface it plugs into), *not* a backend swap (nothing to swap), and *not* a tier composition (it delegates to nothing). |
| **No-op default** | Nothing configures or invokes this package on any `run`/`replay`/`observe`/`calibrate` path. Not-configured is therefore the *only* state that exists after this RFC, and simulator output is byte-identical to before — verifiable by a seeded before/after stdout diff. |

### H2: `sim/internal/hello/format`

| Field | Value |
|---|---|
| **Name** | `sim/internal/hello/format` |
| **Responsibility** | Delimits an already-produced greeting with brackets for presentation. |
| **Surface** | One exported function, `Format`: takes a greeting string and returns that greeting wrapped in a leading `[` and a trailing `]`. It returns only a string — no error — because every input string, including the empty string, has a valid bracketed form. |
| **Allowed imports** | `sim/internal/hello` (the requested dependency); Go stdlib `fmt`. **Denied:** everything else in the repo, including `sim/internal/util` — `format` must reach utility behavior through `hello`, not around it, so the two arrows stay a chain and not a triangle. |
| **Contracts** | **BC-H2-1 (delimiter shape):** GIVEN any input string, WHEN `Format` is called, THEN the result's first character is `[` and its last character is `]`, and the result is at least two characters long. *[evidenced: property_test]*<br>**BC-H2-2 (payload preserved verbatim):** GIVEN any input string, WHEN `Format` is called, THEN removing exactly the first and last characters of the result yields the input unchanged — no trimming, escaping, re-casing, or truncation. *[evidenced: property_test]*<br>**BC-H2-3 (exact length delta):** GIVEN any input string, WHEN `Format` is called, THEN the result's length is exactly the input's length plus two. *[evidenced: property_test]*<br>**BC-H2-4 (nesting is linear, not idempotent):** GIVEN any input and any nesting depth n ≥ 1, WHEN `Format` is applied n times, THEN the result carries exactly n leading `[` and n trailing `]` and the original payload intact. `Format` is explicitly **not** idempotent — this contract fixes that as intended behavior so no caller assumes double-wrapping is a no-op. *[evidenced: metamorphic_test]*<br>**BC-H2-5 (composition with H1 never yields a bare or empty pair):** GIVEN any recipient-name input, WHEN the H1 greeting is formatted, THEN the result is at least three characters long — i.e. never the bare `[]` — which is BC-H1-1 surviving composition. *[evidenced: metamorphic_test]*<br>**BC-H2-6 (purity / determinism):** as BC-H1-5, for `Format`. *[evidenced: property_test]* |
| **Evidence type** | property_test (BC-H2-1, -2, -3, -6), metamorphic_test (BC-H2-4, -5). `differential_test` again not claimed, for the same reason as H1. |
| **Invariants** | **INV-6** only, preserved via BC-H2-6, and vacuously at the run level (no caller on any simulator path). All others not applicable, for the same reasons given in H1. |
| **Extension type** | **tier composition** — a delegation wrapper that adds one presentation concern on top of H1's surface without altering it. This is the taxonomy's best fit for H2 (unlike H1, H2 genuinely delegates), though the taxonomy was written for KV tiers rather than string formatting. |
| **No-op default** | Same as H1: no configuration surface and no caller on any simulator path, so simulator output is byte-identical to before. |

---

## Section 3: Trade-offs & Decisions

| Decision | Alternatives considered | Why this approach | What breaks if this decision is wrong |
|---|---|---|---|
| **`format` is a child package of `hello` and imports its parent** | (a) siblings — `sim/internal/hello` + `sim/internal/helloformat`; (b) one package with both functions | The requested shape, and it reads well: the child is discoverable from the parent and the nesting communicates that `format` is presentation for `hello` specifically. Go permits a child importing its parent. | **This is the one real hazard.** The arrow is only safe while it stays one-directional. If `hello` ever imports `format` (e.g. someone has `Greet` return a pre-bracketed string), the build breaks with an import cycle, and archon `health` loses `cycles: none` — the property the baseline shows the repo holds today. Mitigation: H1's allowed-imports whitelist names `sim/internal/hello/format` as explicitly **denied**, so the denial is reviewable rather than folklore. Fallback if it does go wrong: flatten to alternative (a), which is mechanical. |
| **`hello` depends on `sim/internal/util`, and the surface is shaped so that dependency is real** | (a) single-name `Greet` and no `util` arrow at all; (b) keep the `util` arrow with `Greet` not actually using it | The dependency was requested. `util`'s entire surface is one length-as-int64 helper, so a plain single-name `Greet` would have nothing to call — the arrow would be decorative. Rather than declare an unused import (which Go will not even compile) or fake the arrow, the surface takes zero-or-more recipients and the count it reports is the natural consumer of that helper. Alternative (b) is rejected outright: a whitelist entry nothing exercises is a lie in the contract. | If reviewers judge multi-recipient greeting to be scope the feature does not need, then the honest resolution is alternative (a) — **drop the `util` arrow rather than keep a dead one.** Flagging this for team discussion: the requested dependency is what motivates the multi-recipient surface, not the reverse. Nothing else in the design depends on the outcome. |
| **Non-emptiness is a total guarantee (no error return), not a caller-checked precondition** | (a) return greeting + error for blank input; (b) return empty string for blank input and document it; (c) panic on blank input | The single property most worth owning here is "safe to print". A total function moves that check out of every call site permanently. Blank input is not an error — it has an obvious sensible answer (a generic greeting). Library code panicking on ordinary input would violate the error-handling boundary in `principles.md`. | The caller loses the ability to distinguish "no names" from "all names blank" from the return value alone (recorded above as an accepted simplification). If a real consumer needs that distinction, the fix is a second, explicitly-validating entry point — not weakening BC-H1-1, which is the whole point of the package. |
| **Brackets are hard-coded; no delimiter-strategy interface** | (a) a delimiter parameter; (b) a `Delimiter` interface with bracket/paren/quote implementations | One consumer shape, one delimiter. `principles.md` favors single-method interfaces where a seam is genuinely needed; inventing a strategy seam for a hypothetical second delimiter is speculative generality, and it would freeze a surface no caller exercises. | A second delimiter style becomes a surface change rather than a config change. Cheap: the function is one line and has exactly one contract set, so adding a sibling or a parameter later is a small, well-tested edit — much cheaper than carrying an unused interface. |
| **The packages land with zero call sites** | (a) land the packages *and* convert existing ad-hoc greeting strings in the same change | Zero call sites means zero blast radius and a trivially provable no-op default: nothing on a simulator path can change, so INV-6 byte-identity is unconditional rather than argued. Adoption can then be reviewed per call site. | The risk is the packages sit unused and rot. Accepted for now, but it is the honest weakness of this RFC: **two contract-tested packages with no consumer are speculative until adoption lands.** If the team wants adoption in scope, that is a third hole, not a widening of H1/H2. |
| **`internal/` placement** | a publicly importable `sim/hello` | `internal/` keeps these out of any external API surface while the shape settles, matching the existing `hash`/`kvkey`/`tokenid`/`util` neighbors the baseline shows are the repo's healthy leaves. | External consumers cannot import it. No external consumer is known or wanted. |

---

## Section 4: Delivery Order

```
H1 (sim/internal/hello)        — depends only on existing sim/internal/util; can start first
H2 (sim/internal/hello/format) — depends on H1 (imports it, and BC-H2-5 composes with BC-H1-1)
```

Strictly sequential; there is no parallel opportunity across these two holes. H2's PR targets the same feature branch as H1's and must land after it.

---

## Which components are affected?

- [ ] Core simulator (`sim/`)
- [ ] Cluster simulation (`sim/cluster/`)
- [ ] Workload generation (`sim/workload/`)
- [ ] KV cache (`sim/kv/`)
- [ ] Decision tracing (`sim/trace/`)
- [ ] CLI (`cmd/`)
- [x] New package needed — two: `sim/internal/hello` and `sim/internal/hello/format`

No existing component is modified. `sim/internal/util` is imported but not changed.

## Extension friction check

- **How many files would need to change?** Zero existing files. New files only: one implementation + one test file per hole (4 total). No change to `sim`, `workload`, `cmd`, or any config/trace type — so neither god-module's surface grows.
- **New interface, or extend an existing one?** Neither. Two concrete leaf functions; no interface is defined and none is implemented. There is no existing interface these plug into, which is precisely why they are new packages rather than policy templates.
- **Invariants affected?** Only INV-6 (determinism), and it is preserved by construction: both functions are pure, read no clock/RNG/environment, hold no package state, and iterate no map (R2). Vacuous at the run level since nothing invokes them on a simulator path. Conservation (INV-1), KV conservation (INV-4), causality (INV-5), work-conserving (INV-8), and run/replay parity (INV-13) are not applicable — no request, KV block, event, clock, or trace field is touched.

## Alternatives considered

See Section 3 for the full table with rationale and failure cost.

## Relationship to existing work

No open issue covers this. It follows the existing `sim/internal/` leaf-utility pattern (`hash`, `kvkey`, `tokenid`, `util`) that the archon baseline identifies as the repo's healthiest region. It does not relate to any active epic; nothing in the KV-offload epic (#1585) or the SDLC work depends on it.

---

## Self-Review (8 design review perspectives, per `perspectives.md`)

Performed before opening, as `rfc.md` requires. Findings reasoned about and fixed; the residual items below are recorded rather than hidden.

| # | Perspective | Outcome |
|---|---|---|
| 1 | Motivation & Scoping | **Fixed.** First draft had no modeling-decisions table (arguably N/A for a string utility); replaced with a table of the presentation decisions that *are* contestable, each stating what is lost. Scope-out list made explicit about call-site adoption. |
| 2 | Module Contract Completeness | **Fixed.** All nine contract fields present for both holes. Invariants are named *and* the not-applicable ones are named as not-applicable with a reason, rather than the list being padded with INV-1/4/5 to look thorough. |
| 3 | Extension Framework Fit | **Fixed, with a recorded caveat.** Extension type assigned per hole (H1 subsystem module, H2 tier composition) and the taxonomy strain is stated openly instead of forcing a clean-looking label. No-op default specified for both. Parallel development is **not** possible here (H2 imports H1) and Section 4 says so rather than claiming parallelism. |
| 4 | Trade-off Quality | **Fixed.** Every non-obvious decision has alternatives, rationale, and failure cost. Two are flagged for team push-back rather than presented as settled: the `util` arrow (which drives the multi-recipient surface) and landing with zero call sites. |
| 5 | Validation Strategy | **Fixed.** Every contract carries an evidence type. `differential_test` is explicitly declined for both holes — no oracle exists — instead of being claimed to fill the slot. The no-op default has a stated verification: seeded before/after stdout diff. |
| 6 | Staleness Resistance | **Fixed.** Contracts are GIVEN/WHEN/THEN over observable strings (length, first/last character, substring containment, stated count). No internal field or type name appears in any THEN clause, so the tests they drive should survive a full rewrite (the refactor-survival test in `principles.md`). |
| 7 | Domain Expertise | **N/A, stated as such.** No DES events are introduced, so there is nothing to classify exogenous/endogenous; no clock, no heap, no event queue. No vLLM serving behavior is modeled. No scaling assumption is made. Recording "not applicable" rather than manufacturing a domain angle. |
| 8 | Prohibited Content | **Fixed.** No Go code, no struct or type definitions, no method bodies, no `file:line` references, no file paths for unwritten code. Surfaces are described in prose. Package paths appear only in the contract `Name` field, which the template requires. |

**Residual (neither CRITICAL nor IMPORTANT — for discussion, not blocking):** the `util` arrow shapes H1's surface (Section 3, row 2), and the packages land with no consumer (Section 3, row 5). Both are recorded as open trade-offs; the quality gates below are met either way the team decides.

## Quality Gates

- [x] Every hole has a complete module contract (all fields filled)
- [x] Every non-obvious decision has alternatives + rationale
- [x] No-op default specified — no CLI/config surface and no caller on any simulator path, so output is byte-identical
- [x] Validation strategy specified — INV-6; property_test and metamorphic_test per contract; `differential_test` explicitly declined with reason
- [x] No implementation details for THIS repo — no Go code, struct definitions, or file paths for unwritten code
- [x] Invariants cross-referenced — INV-6 preserved; INV-1/2/4/5/8/12/13 stated not-applicable with reasons
- [x] Extension type identified per hole

## After Agreement

Next step is user-driven: encode into a `.archon` plan + create sub-issues per `docs/contributing/templates/rfc-to-plan.md`. Not started.

---

## Agreed Resolution of the Two Open Trade-offs

Recorded from the maintainer comment on #6 by @mtoslalibu (2026-08-27T16:18:02Z), which is part of the agreed design:

> LGTM, agreed on the design.
>
> Resolving the two trade-offs you flagged for discussion:
>
> 1. **Variadic `Greet` — accepted.** Good catch that `util`'s only export is `Len64`, so a single-name `Greet` would leave the requested arrow decorative and the import unused (which wouldn't compile). Keep `Greet` taking zero or more recipients so `Len64` is genuinely called. The `hello → util` arrow stays.
>
> 2. **Landing with zero call sites — accepted as scoped.** Adoption stays out of scope. The byte-identical no-op default is the property we want here, and the contracts are fully verifiable without a caller.
>
> Contract IDs, hole boundaries, delivery order (H1 then H2, strictly sequential) all agreed as written. Proceed to `.archon` encoding and sub-issues.

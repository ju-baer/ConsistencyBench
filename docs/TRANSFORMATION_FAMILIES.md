# The Five Transformation Families

Every probe pair in ConsistencyBench instantiates one of five formally
defined logical transformations. Together they are documented to cover
roughly 83% of the logical relationships that occur in naturally occurring
question-answering (the remaining ~17% falling to multi-hop reasoning,
modal logic, and quantifier scope — noted as future extensions).

Full machine-readable specs live in
`src/consistencybench/theory.py::FAMILY_FORMAL_SPEC`.

---

## 1. Composition (transitivity) — 18.4% coverage

**Logical basis:** relation composition / transitivity.

**Formal constraint:** `A=>B AND B=>C IMPLIES A=>C`

**Consistency rule:** If a model accepts `A=>B` and is told `B=>C` as
context, it must accept `A=>C`.

**Probe structure:**
- `prompt_a` establishes `A=>B` via a factual claim
- `prompt_b` queries `A=>C`, given `B=>C` as context

**Scoring:** LLM-as-Judge (`ljs`) — the two responses are free-form and the
composed conclusion isn't reducible to a yes/no string match.

**Why it's hard:** composition requires the model to *chain* information
across two independent responses rather than restate a fact — testing
whether logical structure survives being split across separate turns.

---

## 2. Reversal (symmetric relations) — 14.2% coverage

**Logical basis:** reversal of a symmetric binary relation.

**Formal constraint:** `rel(X,Y) <=> rel(Y,X)` (for relations symmetric by
definition — e.g. "is married to", "is the same distance from")

**Consistency rule:** answers to `rel(X,Y)` and `rel(Y,X)` must be the same.

**Probe structure:**
- `prompt_a` queries `rel(X,Y)`
- `prompt_b` queries `rel(Y,X)`

**Scoring:** Rule-Based Scoring, rule = `same` (`config.RBS_RULES["reversal"]
== "same"`) — both answers are designed to extract as a clean yes/no.

**Why it's hard:** surface word order changes (X-then-Y vs. Y-then-X) can
trigger spurious recency/primacy effects even though the underlying relation
is symmetric by construction.

---

## 3. Complement (negation) — 28.6% coverage, the largest family

**Logical basis:** truth complement / negation.

**Formal constraint:** `NOT(assert(P) AND assert(NOT-P))`

**Consistency rule:** a model cannot affirm `P` in one response and also
affirm `NOT-P` in the other.

**Probe structure:**
- `prompt_a` asserts (or asks the model to evaluate) a proposition `P`
- `prompt_b` asks whether `NOT-P` is acceptable

**Scoring:** Rule-Based Scoring, rule = `opposite`
(`config.RBS_RULES["complement"] == "opposite"`) — consistency means the two
extracted yes/no answers *differ*.

**Why it's the largest family and why it matters most:** this is the most
basic form of self-contradiction — affirming a claim and its negation in the
same conversation. Its high real-world prevalence (28.6%) is part of why it
shows up as one of the two hardest families across nearly every evaluated
model (the other being Equivalence) in the reference run.

---

## 4. Ordering (asymmetric temporal relations) — 10.8% coverage

**Logical basis:** asymmetric temporal ordering.

**Formal constraint:** `before(A,B) <=> after(B,A)`

**Consistency rule:** answers to "did A come before B" and "did B come after
A" must be the same.

**Probe structure:**
- `prompt_a` queries whether A came before B
- `prompt_b` queries whether B came after A

**Scoring:** Rule-Based Scoring, rule = `same`
(`config.RBS_RULES["ordering"] == "same"`).

**Why it's hard:** unlike Reversal, the relation itself is *asymmetric*
(before ≠ after), so consistency requires correctly inverting the temporal
frame rather than just recognizing symmetry — a subtly different cognitive
operation despite the similar surface structure to Reversal.

---

## 5. Equivalence (semantic paraphrase) — 11.4% coverage

**Logical basis:** semantic equivalence preservation.

**Formal constraint:** `equiv(pA,pB) IMPLIES compat(rA,rB)`

**Consistency rule:** logically equivalent prompts must yield compatible
responses, even when their surface form is maximally different.

**Probe structure:**
- `prompt_a` is one surface form of a proposition `phi`
- `prompt_b` is a second surface form of the *same* `phi`, with the
  lexical/semantic/syntactic distance between the two forms deliberately
  maximized (via the `delta` targeting described in
  [METHODOLOGY.md](METHODOLOGY.md)) while truth-conditional content is held
  fixed

**Scoring:** LLM-as-Judge (`ljs`) — with no fixed relational structure to
exploit, only a judge that understands both phrasings can assess whether the
two responses are compatible.

**Why it's hard:** this family is difficulty-calibrated by construction —
`prompt_a` and `prompt_b` are designed to be as different in wording as
possible while remaining truth-conditionally identical, directly probing
whether "understanding" survives a change of surface form. Alongside
Complement, this is consistently one of the two hardest families in the
reference run.

---

## Design rationale: why these five and not others

The families were chosen to span:

- **Symmetric vs. asymmetric relations** (Reversal vs. Ordering) — testing
  whether models track relation directionality correctly rather than pattern
  -matching on symmetry.
- **Structural vs. surface-form transformations** (Composition/Ordering vs.
  Equivalence) — testing logical chaining separately from paraphrase
  robustness.
- **Single-proposition vs. two-proposition constraints** (Complement vs. the
  rest) — testing the most basic form of self-contradiction distinctly from
  cross-statement inference.
- **Rule-based vs. judge-based verifiability** — three families (Reversal,
  Complement, Ordering) admit fully mechanical, judge-free scoring, serving
  as an anchor against which the LLM-as-Judge families (Composition,
  Equivalence) can be validated (see the inter-annotator agreement
  methodology in [METHODOLOGY.md](METHODOLOGY.md#5-scoring-rbs--ljs-ensemble)).

The remaining ~17% of real-world logical relationships not covered by these
five families — multi-hop reasoning chains, modal operators ("must",
"might"), and quantifier scope ambiguity — are noted as natural extensions
for future transformation families, but were left out of the initial release
to keep each family's formal constraint unambiguous and its scoring method
well-validated.

This repository is the first concrete step in a longer research program: characterizing the relationship between a language model's residual stream and the space of completions it induces, using the conceptual machinery of abstract interpretation — adapted honestly to a setting where its classical guarantees do not hold.
The toy experiments here (Z1R, Mess3, small transformers) are calibration instruments, not the subject. The subject is the method. If you are an agent picking this up, your job is to extend the method toward systems where there is no ground truth, while preserving the property that made the toy phase valuable: every claim was checkable, and every failure was typed.
The intellectual frame (hold these commitments)

The concrete semantics of a prefix is its completion measure. A residual state is a candidate abstract value whose concretization is a probability kernel over futures, not a set. This breaks the classical Galois-connection setup at the root. Do not paper over this; build on its replacements.
Completeness is real-valued, directional, and relative. There is no binary "sound/complete" here. There is a divergence (how much the abstraction's predicted completion measure differs from the true one), and it is always indexed by three things: the tolerance policy, the interpreter class (V-information: what a bounded probe family can extract), and the horizon/distribution over which completions are measured. Any result that omits one of these indices is overclaiming. We learned this the hard way more than once.
Sufficiency, not bisimulation, is the static notion. Two residuals are equivalent when they induce (nearly) the same completion measure. Bisimulation-style coherence re-enters only when you demand consistency under autoregressive extension — a legitimate v-next, not the default.
The ideal abstraction is the belief state (the minimal sufficient statistic for the future; computational mechanics' mixed-state presentation). Where ground truth exists, discovered abstractions are judged against it. Where it doesn't, the belief state is the regulative ideal the diagnostics approximate.
The complete shell is of the realized semantics, not the ambient one. The single most surprising empirical lesson so far: the loop correctly found a 1-D shell for a process whose ambient belief simplex is 2-D, because the reachable belief set was three points. Abstractions answer to what the system actually does on the distribution it actually sees.

The methodology that earned its keep

The outer loop is CEGAR: propose an abstraction, test its quantified completeness, mine counterexamples, refine or coarsen, stop at a declared fixed point. Its real output is not the fixed point but the taxonomy of typed failures, each with an observable, ground-truth-free diagnostic:

domain coarseness — conflated pairs with divergent completions;
interpreter incompleteness — decode fails but a nonparametric decode succeeds (the V-information gap);
metric junk-domination — counterexamples persist under refinement while decoding already suffices;
proposal misalignment — the structure exists in the full representation but not in the proposed subspace (variance is not relevance);
vacuous tolerance — the no-information baseline already passes. Always compute the baseline.


Honesty constraints are load-bearing. Proposal families and probes may be supervised only on observables (completions); privileged ground truth (beliefs, hidden states) is evaluation-only. Pre-register predictions before runs. Construct adversarial caches with known answers to validate every code path and verdict branch — several genuine design flaws were caught only this way.
Report curves, not points. The complete shell is a staircase k*(tol); a single fixed point is one vertical line a reader should be free to move. Stopping rules must be declared, with explicit margins.

Where this is going (roadmap, in order)

Interventional upgrade. Replace correlational sufficiency with causal abstraction: interchange interventions on the discovered subspaces, scored behaviorally. This is the natural translation of the completeness interchange law and the strongest claim available without ground truth.
Coherence under generation. Test whether discovered abstractions update consistently under autoregressive extension (the bisimulation condition) — required before treating them as state, not just summary.
Richer processes, weakening ground truth gradually. PCFGs (exact via inside–outside), then Dyck/stack languages, then TinyStories-scale models where ground truth becomes sampled rather than exact. At each step, verify the diagnostics still rank abstractions correctly while the oracle is still available to check them.
Real LLMs. No oracle. What transfers is: baseline calibration, the failure taxonomy, capacity-indexed completeness curves, behavioral-divergence metrics, supervised-on-observables proposal families (including SAE features as proposals, evaluated rather than trusted), and interventional tests. The deliverable is measured missability: for a given concept, distribution, probe class, and tolerance, how much completion-relevant structure escapes — with error bars.

Scope honesty (do not let this erode)
This method sees what is load-bearing for completions on the evaluation distribution. Concepts that rarely move the completion measure — including precisely the rare, normative, or adversarially hidden properties safety cares most about — are structurally invisible to it on-distribution. Passing all tests certifies "functions as a sufficient summary under these probes and this distribution," never identity with a human ontology, and never off-distribution guarantees. The program's honest pitch is not "we can find everything"; it is "we can quantify what is findable and state what is not."
Working norms for agents
Keep everything runnable on modest hardware with exact or cheap ground truth as long as possible. Quarantine heavy dependencies. Make analysis stages consume caches, not models. When a result surprises you, suspect the harness before the science — then, if the harness holds, write the surprise into the verdict logic so the code can diagnose it next time without you. Prefer adding a typed verdict over adding a number. And when you fix a flaw, document the flaw and its lesson where the fix lives; the failures are the curriculum.

Library home, not frozen imports (forward rule, added after the oracle-withdrawal arc). Shared infrastructure — `build_candidates`, the observable ρ helper (`rho_obs`/`_mnorm`/`rho_band`), registered constants, and any verdict-partition helper — belongs in the living library (`battery.py` / `expcommon.py`), and concluded/frozen scripts import *from* the library, never the reverse. The oracle-withdrawal arc violated this: exp 24's `reference_selection.py` (frozen) defined `build_candidates` inline and exps 25–27 imported it, and `rho_obs` got hand-copied three times (the copies already drifted). A frozen record must not be load-bearing infrastructure. **First step of any next phase:** promote that machinery into the library and import it there; capture the recalibrate/inversion verdict pattern (FORMALISM §6.1 rule 9) as a shared `partition()` helper. Promote forward — do not retro-edit the frozen scripts.

Experiment reviews
Use `EXPERIMENT_REVIEW_PROTOCOL.md` for pre-registration and result
reviews. These reviews are not generic code review. They evaluate whether
the experiment is the right one for the research plan, whether the code
implements the registered construct, and whether verdicts and conclusions
stay within what was measured. Review pauses are part of the method, not
administrative overhead.

Pre-registration is a two-part artifact: the experiment writeup and the
code that implements it. The writeup must state the goals, assumptions,
scope, predictions, adjudication rules, and reviewable failure modes; the
script must already exist and implement those rules, guards, self-checks,
and output tables. Pause after committing a pre-registration, before the
first run, for review of both the experimental design and the code. Pause
again after running, before writing conclusions, for review of the results,
verdict logic, LLM-work creep, and maintainability regressions. Do not
treat "pre-registered" as complete if either the writeup or the runnable
implementation is missing.

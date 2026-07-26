# Development timeline and experimental learnings

This is a chronological research narrative distilled from the project's
append-only experiment log. Version numbers refer to compact lane checkpoints
or pinned evaluation stages, not releases of the PersonaPlex base model.

The purpose of this timeline is reproducibility of **reasoning**, not merely
reproducibility of the final command: what hypothesis was tested, what evidence
failed, and why the next experiment changed direction.

## Phase 0 — Define native agentic duplex behavior

**Starting point.** The project began from the research direction described by
[`hyzhang24/DuplexSLA`](https://github.com/hyzhang24/DuplexSLA): user speech,
assistant speech, and actions should advance on one conversational clock.

**Decision.** Use PersonaPlex as the available duplex backbone, but do not wrap
it in ASR → planner → TTS. Require the model itself to emit calls and speak
after receiving results in its continuing state.

**Implementation difference.** DuplexSLA describes a textual action stream on
a 160 ms Step-Audio-2-mini timeline. This project created typed five-slot
action and environment lanes on PersonaPlex/Moshi's 80 ms clock.

**Lesson.** “Native tool calling” must describe ownership and causality, not
only low latency. A fast external planner is still an external planner.

## Phase 1 — Build the protocol before training

**Hypothesis.** Tool traffic needs a strict symbolic contract before neural
predictions can safely cause side effects.

**Technique.** Implemented five micro-slots per frame, action grammar masking,
typed environment events, `REF` leases, call UUIDs, terminal receipts, causal
FIFO serialization, and future-result leakage checks. Added fixed schemas for
weather, timer, and home tools.

**Result.** Dependency-light CPU tests established packet validity, causal
visibility, stale-result rejection, cancellation, timeout, and single-release
reference behavior before the 7B model was involved.

**Lesson.** Neural action quality and executor safety are separate problems.
The runtime must reject malformed or unauthenticated actions even when the
model is otherwise coherent.

## Phase 2 — Prove the real 7B mechanical loop

**Hypothesis.** PersonaPlex could carry additional action/environment state
without breaking its audio path.

**Technique.** Added compact lane modules to the Moshi generator, forced a
known action packet for diagnosis, injected a later executor observation, and
reloaded lane-only checkpoints without duplicating the base weights.

**Result.** The real model accepted five-slot actions, consumed a later typed
observation in the same state, continued audio generation, saved a compact
checkpoint, and reloaded it successfully.

**Lesson.** Forced tokens prove wiring, shapes, and causality only. They are not
evidence that speech semantics caused the model to call a tool.

## Phase 3 — First autonomous weather-call proof

**Hypothesis.** A deliberately overfit lane could autonomously reproduce one
model-owned call under causal speech input.

**Technique.** Trained the narrow weather checkpoint on “What's the weather in
Makarska?” and removed forced actions, logit bias, and external intent logic.

**Result.** The model emitted `weather.lookup(REF_0, CITY_14)` and received the
mock result in its continuing state. Later prompted, held-out, and microphone
traces produced premature calls, repeated calls, and wrong-schema packets.

**Lesson.** An autonomous call is stronger than a forced call, but one overfit
trajectory still demonstrates mechanics rather than robust speech
understanding. Placeholder IDs can conceal semantic failure.

## Phase 4 — v35 and the first protected narrow action policy

**Hypothesis.** Separate call-start detection from packet content and protect a
no-tool control while learning several tool schemas.

**Technique.** Added prompted feature extraction, runtime grammar masks,
scenario-derived seeds, a calibrated packet-start margin, and held-out HOME,
WEATHER, TIMER, and no-tool evaluation.

**Result.** v35 became a safe but narrow baseline: strong call-start behavior,
one exact unseen HOME packet, coherent audio/text, and a clean no-tool control,
but weak exact-packet coverage across the full held-out slice.

**Lesson.** A promotion gate needs both positive tool cases and negatives.
Detection accuracy can look good while argument selection remains poor.

## Phase 5 — v40–v44 augmentation and branch separation

**Hypothesis.** More voice and scenario coverage could broaden the action lane
without erasing v35's reliable call-start boundary.

**Technique.** Generated balanced Microsoft David/Zira SAPI variants, audited
training shards, tried regularized augmentation, and finally separated the
frozen start branch from newly trained packet-content tensors.

**Result.** Straight augmentation provided little exact-packet improvement and
sometimes regressed starts. The v44 dual-branch design preserved v35's start
behavior while allowing content work to continue.

**Lesson.** Catastrophic interference can happen inside a small adapter. Freeze
the sub-behavior that already passes and train new capacity for the failing
sub-behavior.

## Phase 6 — v45–v57 result-grounded speech attempts

**Problem.** Correct calls and visible results did not reliably produce spoken
answers grounded in those results; weather often continued a denial already
started before the result arrived.

**Techniques attempted.** Response-only adapters, word-timed targets, temporal
recurrent state, assistant-prefix roll-ins, multiple teacher-forced epochs,
within-response scheduled sampling, a model-call-to-result pending-speech hold,
and tool-balanced held-out evaluation.

**What improved.** Teacher-forced NLL, top-1 response tokens, and causal value
of the environment result improved substantially. Pending-aligned training let
result information enter free generation.

**What failed.** WEATHER still denied, collapsed into numeric loops, or emitted
malformed speech. Better teacher-forced scores repeatedly failed to predict a
usable closed-loop answer. A response adapter could not undo words the base
model had already committed before seeing the result.

**Lessons.** Closed-loop exposure matters more than isolated target likelihood.
The model's sampled text and audio are part of the next causal state. Training
must match that state, and speech may need to remain pending until a tool result
is visible.

## Phase 7 — v58 first clean result-grounded native turn

**Hypothesis.** A model-owned sentence boundary could terminate a grounded turn
without a host-authored response or arbitrary timeout.

**Technique.** Added a post-result sentence latch that activates on the model's
own punctuation, preserves the generated sentence, and supplies native silence
afterward while user/action/environment lanes continue.

**Result.** On held-out HOME, v58 emitted one exact temperature call, held
speech through result arrival, observed the typed result, said `Okay, keeping
it 30.`, and terminated cleanly. WEATHER still called correctly but spoke an
incorrect denial; another TIMER case chose the wrong tool.

**Lesson.** Turn termination and semantic grounding are independent. A clean
latch can stop a response but cannot make wrong content truthful.

## Phase 8 — v62–v70 exact TIMER routing

**Hypothesis.** TIMER failures were localized to content/argument routing and
could be repaired without changing the protected start policy.

**Technique.** Scoped training to timer tool and duration rows, audited every
checkpoint delta, and used a frozen timer utterance plus exact packet verifier.

**Result.** v70 produced the first exact held-out
`timer.create(REF_0, MIN_2)` mechanics trajectory. The result entered the same
state, but the native speech was not yet a reliable confirmation.

**Lesson.** Verify call mechanics and response grounding separately. A correct
tool packet does not imply the spoken model used the returned result.

## Phase 9 — v71–v106 TIMER response grounding

**Techniques.** Timer-specific response training, pending holds, sentence
termination, response-token context, concise targets, roll-ins, and repeated
seed-pinned closed-loop diagnostics.

**Result.** Intermediate checkpoints produced blanks, repetitions, duration
loops, misspellings, or correct calls without useful speech. v88/v94 delivered
narrow grounded timer behavior with rough output. v106 produced an exact call,
visible result, finite audio, and semantically grounded wording containing
`set`, `2`, and `minutes`, while omitting the literal word `timer`.

**Lesson.** Semantic acceptance and presentation quality should be separate
metrics. Tokenized evidence prevents substring errors such as treating `20` as
the requested duration `2`, while a separate literal-word flag preserves the
higher quality bar.

## Phase 10 — v109–v151 repair sweeps

**Hypothesis.** Carefully scoped row repairs could make the grounded response
stable across seeds while protecting action and audio behavior.

**Technique.** Built response-repair aggregates, environment/onset scopes,
nonnegative row constraints, transition margins, and strict tensor-delta and
held-out authorization gates. Hundreds of teacher-forced comparisons narrowed
which rows could change safely.

**Result.** Many candidates improved supervised metrics but regressed another
tool, produced blank speech, looped, chose malformed continuations, or failed
closed-loop sampling. v151 combined a safer onset and post-onset suppression,
passed static protection checks, but was rejected by native sampling.

**Lesson.** Teacher-forced authorization is necessary for damage control, not
sufficient for promotion. The decisive test is the model's own sampled causal
trajectory.

## Phase 11 — v160–v163 causal proposal and prefix work

**Hypothesis.** Once a valid first result-grounded semantic proposal appears,
holding it causally could prevent the decoder from drifting immediately.

**Technique.** Added a default-off semantic proposal latch, captured full
continuation traces, built causal-prefix repair shards, and compared multiple
seeds without changing tool mechanics.

**Result.** v160 made all four responses begin with the intended grounded
onset, but later tokens still diverged into malformed continuations. v161
showed the failures were reproducible, and v162/v163 isolated prefix repair
without safely solving the whole continuation.

**Lesson.** Fixing the first token is not equivalent to fixing the response.
The sampled assistant-audio history changes future text state and must be
represented in training.

## Phase 12 — v164–v169 continuation and dual-state training

**Hypothesis.** A dedicated post-onset continuation residual trained against
both original and newly sampled states could repair later tokens while leaving
the call, result, first grounded token, and audio predictions protected.

**Technique.** Added a rank-32 response continuation branch, trained only on
authorized post-`set` transitions, then constructed v168 with both v160 and
v167 causal histories. v169 trained from frozen v151 on all 40 dual-state
transitions.

**Result.** v169 ranked all 40 continuation targets first, changed only the
authorized residual tensors/rows, preserved 10,096 held-out audio top-1
predictions, and introduced no previously-correct text regression. Four sampled
rollouts were still stylistically unstable. Native text top-k 1 made all four
finite and grounded, leading to the v172 full known-WAV qualification with one
exact call, one visible result, and non-silent native audio.

**Lesson.** Train on the states created by the candidate itself, not only its
parent. Sampling policy is part of a reproducible model trajectory and must be
recorded, but it must not be confused with forced tool tokens.

## Phase 13 — Replace offline recording with a persistent session

**Problem.** The first “video demo” launcher recorded a complete utterance,
then started inference. It was not the continuous conversation the project was
supposed to demonstrate.

**Technique.** Built `live_session_v2.py` to keep Mimi, PersonaPlex, action
grammar, typed executor observations, and Mimi decoding alive on one 80 ms
clock. Added acoustic turn windows only to bound when `CALL_BEGIN` is legal;
the VAD does not select the tool.

**Result.** The corrected persistent causal replay made v169 emit exactly one
`timer.create(MIN_2)` at frame 20, schedule a real 120-second timer, observe
`OK` at frame 25 in the same state, and generate `Sure, set a 2 minutes.` plus
finite audio. No repeated call occurred.

**Lesson.** Architecture claims must be tested in the intended session
lifecycle. Record-then-infer success cannot stand in for a continuously loaded
duplex agent.

## Phase 14 — Microphone attempts and invalid evidence

**First USB-microphone result.** PersonaPlex generated coherent speech but no
action packet or tool result. This was a real negative, not a demo success.

**Second attempt.** A persistent replay stopped on the wall-clock limit while
still processing long leading silence and never reached the measured speech.
That no-call trace was marked invalid rather than counted as model failure.

**Corrections.** Excluded model loading from the session limit, removed the
finite-replay wall-clock cutoff, measured speech boundaries, preserved the USB
recording for later Linux replay, and stopped presenting the legacy launcher as
live.

**Lesson.** Failed, invalid, and negative results are different categories. A
run that never reaches the stimulus cannot evaluate tool semantics.

## Phase 15 — Windows performance and rejected acceleration

**Measurement.** The valid 46-frame v169 persistent trace ran at real-time
factor 12.991. Mean Mimi encode was 46.67 ms, model step 965.38 ms, and Mimi
decode 26.53 ms; the 7B step dominated latency.

**Technique.** Tried full, backbone-only, and audio-only Python CUDA graph
capture while keeping the action sampler, grammar, executor, and observations
outside the graph.

**Result.** Some timings improved, but every graph variant changed behavior:
empty graphs, silent audio, and no call/text. All were rejected despite speed
improvements. Eager v169 remains the semantic reference.

**Lesson.** Optimization requires a behavioral parity gate. A faster run that
loses the native action or spoken result is a failed optimization, not progress.

## Phase 16 — Why Linux is next

Windows PyTorch lacked the optimized attention path needed for the target frame
budget. The next plan is native Ubuntu: first reproduce the exact eager v169
call/result/text/audio trajectory, then enable optimized attention and compare
semantics before measuring speed. Saved-microphone replay, greeting-then-timer,
no-tool negatives, and only then a live microphone session follow.

**Lesson.** Change one axis at a time. Environment migration must establish
exact parity before compilation, and compilation must establish parity before
claiming real-time readiness.

## Consolidated reusable lessons

1. **Protocol first.** Grammar, schemas, leases, and causal result scheduling
   are safety prerequisites, not cleanup work.
2. **Separate proof levels.** Forced wiring, autonomous call mechanics, result
   visibility, grounded speech, held-out generalization, microphone robustness,
   and real-time performance are different milestones.
3. **Closed loop wins.** Teacher-forced loss can improve while sampled speech
   gets worse; promote only from unforced causal trajectories.
4. **Protect successes.** Freeze working branches and audit exact tensor deltas
   when repairing one failure mode.
5. **Train on candidate state.** Sampled text and audio alter the next state;
   exposure and dual-state training matter.
6. **Do not let the host fake semantics.** A host may validate and execute, but
   should not decide the tool or author the model's confirmation.
7. **Treat timing as semantics.** Results must not appear early, and premature
   speech can make later truth impossible to express.
8. **Keep invalid evidence out.** Runs that stop before the stimulus or use the
   wrong session architecture do not answer the research question.
9. **Require parity for speed work.** Optimization is accepted only when the
   exact action/result/text/audio behavior survives.
10. **Report the boundary honestly.** The current release is a narrow timer
    proof, not a general tool-calling duplex model.

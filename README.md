# PersonaPlex Native Tool Use

Research code and a reproducible narrow checkpoint showing a native tool call
inside a continuing PersonaPlex full-duplex speech-model state.

The promoted proof takes the causal speech request **“Count down two minutes
for me.”** and produces one learned `timer.create(MIN_2)` action. An allow-listed
executor schedules 120 seconds, injects an `OK` observation into the same model
state, and PersonaPlex generates its own text and audio confirmation.

This is not an ASR → planner → TTS pipeline. There is no transcript-driven
intent parser, host-generated tool JSON, host-authored confirmation, or external
TTS in the verified trajectory.

## Research origin and credit

This project was motivated by
[`hyzhang24/DuplexSLA`](https://github.com/hyzhang24/DuplexSLA), which frames
full-duplex agency as synchronized **Speech, Language, and Action** on one
conversational clock instead of placing tool use in an external cascade. We
used that framing to ask a practical follow-up question: can the same native
action principle be adapted to an already available PersonaPlex/Moshi duplex
backbone?

This repository is an independent PersonaPlex implementation, not the official
DuplexSLA code or a reproduction of its Step-Audio-2-mini model. DuplexSLA uses
a textual action channel on a 160 ms chunk timeline; this project uses typed
five-slot action and environment lanes on PersonaPlex's 80 ms frame clock,
including authenticated references and causal executor-result injection. See
[ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for the full attribution and citation.

## Status

What is verified:

- one native `timer.create` call at frame 20;
- arguments `{"minutes":"MIN_2"}`;
- one terminal `OK` result visible at frame 25;
- a real 120-second timer scheduled by the allow-listed executor;
- PersonaPlex-generated text: `Sure, set a 2 minutes.`;
- finite assistant audio;
- a 312 MiB lane-only checkpoint, separate from the 7B base model.

What is not yet verified:

- general-purpose tool selection;
- robust unseen speech or microphones;
- stable sampling across seeds;
- continuous real-time conversation;
- production safety or unrestricted tool execution.

The verified Windows eager run used controlled causal WAV replay and measured a
real-time factor of `12.991`. Treat this as a mechanical and same-state
result-awareness proof, not a finished voice assistant.

## Download the checkpoint

The adapter is hosted on Hugging Face:

[`lazybutai/personaplex-native-tool-use`](https://huggingface.co/lazybutai/personaplex-native-tool-use)

```powershell
hf download lazybutai/personaplex-native-tool-use `
  micro-head-v169-v151-dual-state-margin05.safetensors `
  --local-dir .\checkpoints
```

You must separately obtain
[`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1)
and accept NVIDIA’s model license. This project does not redistribute the base
weights.

## Repository map

```text
src/personaplex_agent/       Typed action/environment protocol and runtimes
personaplex_moshi_patch/     PersonaPlex/Moshi lane-model extensions
scripts/live_session_v2.py  Persistent replay/microphone runner
scripts/verify_v169_trace.py Portable verifier for a newly generated trace
schemas/                     JSON protocol and acceptance contracts
tests/                       Selected dependency-light contract tests
examples/timer-v169/         Frozen input, reference output, proof summary
docs/TECHNICAL_REPORT.md     Architecture, evidence, failures, limitations
docs/DEVELOPMENT_TIMELINE.md Experiment sequence, decisions, and lessons
docs/REPRODUCE.md            Exact Windows reproduction procedure
```

## Quick validation without a GPU

```powershell
python -m pip install -e . pytest
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q tests
python .\scripts\verify_v169_trace.py `
  .\examples\timer-v169\proof-summary.json `
  --proof-summary
```

## Full checkpoint reproduction

See [docs/REPRODUCE.md](docs/REPRODUCE.md). The frozen run requires two CUDA
devices in the demonstrated Windows configuration: Mimi on `cuda:1` and the 7B
model on `cuda:3`. GPU 2 is not used by the documented command.

## Safety boundary

The included executor accepts only `timer.create`, validates the typed duration,
limits it to 1–60 minutes, and rejects unsupported tools. Do not replace this
with unrestricted shell execution. A neural action must be parsed, schema
checked, allow-listed, authenticated to its reference lease, and causally
returned to the model before any side effect is trusted.

## License

Project code is MIT licensed. PersonaPlex base weights are governed by the
NVIDIA Open Model License Agreement. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

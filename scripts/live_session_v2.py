"""Persistent native PersonaPlex microphone/tool session.

Unlike ``talk_tool_v2.py``, this runner never records a complete utterance and
then starts inference.  Mimi encoding, AgentMicroLMGen state, action grammar,
executor observations, and Mimi decoding remain live on the same 80 ms clock
for the entire session.  The host performs only acoustic VAD, protocol safety,
and allow-listed execution; it does not transcribe or plan the model's call.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
from typing import Any, Iterable, Iterator, Sequence
import wave


# Windows PyTorch cannot compile Moshi's optional Triton kernels.  This must be
# set before importing any Moshi module so its wrappers select eager execution.
os.environ.setdefault("NO_TORCH_COMPILE", "1")


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
FORK = ROOT / "fork" / "personaplex-agent-moshi" / "moshi"
for candidate in (str(SRC), str(FORK)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


RUNNER_VERSION = "ppx-native-live-session-v2.1.0"
CHECKPOINT_SHA256 = (
    "f9c5db254f8fea477155a9233b063b03248605c07e95efb987d88bc3626cdcba"
)
FRAME_RATE = 12.5
SAMPLE_RATE = 24_000
FRAME_SIZE = 1_920


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def rms(samples: Any) -> float:
    import numpy as np

    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not values.size:
        return 0.0
    return float(np.sqrt(np.mean(values * values)))


@dataclass(slots=True)
class AcousticTurnWindow:
    """Pure acoustic turn boundary used only to bound CALL_BEGIN timing."""

    threshold: float = 0.008
    min_speech_frames: int = 2
    end_silence_frames: int = 2
    call_window_frames: int = 24
    speech_run: int = 0
    silence_run: int = 0
    in_speech: bool = False
    call_window_remaining: int = 0
    utterance_count: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("VAD threshold must be between zero and one")
        for name in (
            "min_speech_frames",
            "end_silence_frames",
            "call_window_frames",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    @property
    def call_window_open(self) -> bool:
        return self.call_window_remaining > 0

    def update(self, frame_rms: float) -> str | None:
        if not math.isfinite(frame_rms) or frame_rms < 0:
            raise ValueError("frame RMS must be finite and non-negative")
        if frame_rms >= self.threshold:
            self.speech_run += 1
            self.silence_run = 0
            self.call_window_remaining = 0
            if not self.in_speech and self.speech_run >= self.min_speech_frames:
                self.in_speech = True
                return "speech_start"
            return None

        self.speech_run = 0
        if not self.in_speech:
            return None
        self.silence_run += 1
        if self.silence_run < self.end_silence_frames:
            return None
        self.in_speech = False
        self.silence_run = 0
        self.utterance_count += 1
        self.call_window_remaining = self.call_window_frames
        return "speech_end"

    def advance(self) -> None:
        if self.call_window_remaining > 0:
            self.call_window_remaining -= 1


@dataclass(frozen=True, slots=True)
class ToolExecution:
    ok: bool
    payload_tokens: tuple[str, ...]
    human_text: str
    scheduled_seconds: float | None = None


class TimerExecutor:
    """Allow-listed real timer executor for model-emitted TIMER packets."""

    _MINUTES = re.compile(r"^MIN_(\d+)$")

    def __init__(self) -> None:
        self._timers: list[threading.Timer] = []

    def execute(self, tool_name: str, arguments: dict[str, str]) -> ToolExecution:
        if tool_name != "timer.create":
            return ToolExecution(False, ("ERR_EXECUTOR",), "unsupported tool")
        value = arguments.get("minutes", "")
        match = self._MINUTES.fullmatch(value)
        if match is None:
            return ToolExecution(False, ("ERR_SCHEMA",), "invalid timer duration")
        minutes = int(match.group(1))
        if minutes < 1 or minutes > 60:
            return ToolExecution(False, ("ERR_SCHEMA",), "timer duration is outside 1..60")
        seconds = float(minutes * 60)

        def finished() -> None:
            print(f"\nTIMER FINISHED: {minutes} minute(s).", flush=True)

        timer = threading.Timer(seconds, finished)
        timer.daemon = True
        timer.start()
        self._timers.append(timer)
        return ToolExecution(
            True,
            ("ENUM_0",),
            f"timer is set for {minutes} minute(s)",
            seconds,
        )


class LiveSoundDevice:
    """Non-blocking 24 kHz microphone capture and assistant playback."""

    def __init__(
        self,
        *,
        input_device: int | str | None,
        output_device: int | str | None,
        playback: bool,
        max_buffer_frames: int = 2048,
    ) -> None:
        self.input_device = input_device
        self.output_device = output_device
        self.playback = playback
        self.input_queue: queue.Queue[Any] = queue.Queue(maxsize=max_buffer_frames)
        self.output_queue: queue.Queue[Any] = queue.Queue(maxsize=max_buffer_frames)
        self.input_statuses: list[str] = []
        self.output_statuses: list[str] = []
        self.dropped_input_frames = 0
        self._input_stream: Any = None
        self._output_stream: Any = None
        self._output_tail: Any = None

    def __enter__(self) -> "LiveSoundDevice":
        import numpy as np
        import sounddevice as sd

        def input_callback(indata: Any, frames: int, timing: Any, status: Any) -> None:
            del timing
            if status:
                self.input_statuses.append(str(status))
            if frames != FRAME_SIZE:
                self.input_statuses.append(f"unexpected input block size {frames}")
            block = np.asarray(indata[:, 0], dtype=np.float32).copy()
            try:
                self.input_queue.put_nowait(block)
            except queue.Full:
                self.dropped_input_frames += 1

        def output_callback(outdata: Any, frames: int, timing: Any, status: Any) -> None:
            del timing
            if status:
                self.output_statuses.append(str(status))
            outdata.fill(0)
            cursor = 0
            while cursor < frames:
                if self._output_tail is None or not self._output_tail.size:
                    try:
                        self._output_tail = self.output_queue.get_nowait()
                    except queue.Empty:
                        break
                take = min(frames - cursor, int(self._output_tail.size))
                outdata[cursor : cursor + take, 0] = self._output_tail[:take]
                self._output_tail = self._output_tail[take:]
                cursor += take

        self._input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            device=self.input_device,
            channels=1,
            dtype="float32",
            callback=input_callback,
        )
        self._input_stream.start()
        if self.playback:
            self._output_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                blocksize=FRAME_SIZE,
                device=self.output_device,
                channels=1,
                dtype="float32",
                callback=output_callback,
            )
            self._output_stream.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()
        if self._output_stream is not None:
            self._output_stream.stop()
            self._output_stream.close()

    def read(self, timeout: float = 0.25) -> Any | None:
        try:
            return self.input_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def play(self, pcm: Any) -> None:
        if not self.playback:
            return
        try:
            self.output_queue.put_nowait(pcm)
        except queue.Full:
            self.output_statuses.append("assistant playback queue full")

    @property
    def buffered_input_frames(self) -> int:
        return self.input_queue.qsize()

    def drain_playback(self, timeout: float = 5.0) -> None:
        if not self.playback:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            tail_empty = self._output_tail is None or not self._output_tail.size
            if self.output_queue.empty() and tail_empty:
                return
            time.sleep(0.05)


def _coerce_device(value: str | None) -> int | str | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def list_audio_devices() -> int:
    import sounddevice as sd

    default_input, default_output = sd.default.device
    rows = []
    for index, device in enumerate(sd.query_devices()):
        if int(device["max_input_channels"]) < 1 and int(device["max_output_channels"]) < 1:
            continue
        rows.append(
            {
                "index": index,
                "name": str(device["name"]),
                "input_channels": int(device["max_input_channels"]),
                "output_channels": int(device["max_output_channels"]),
                "default_input": index == default_input,
                "default_output": index == default_output,
            }
        )
    print(json.dumps({"audio_devices": rows}, indent=2))
    return 0


def load_wav_24k(path: Path) -> Any:
    import numpy as np
    import sphn

    pcm, source_rate = sphn.read(str(path))
    pcm = sphn.resample(pcm, src_sample_rate=source_rate, dst_sample_rate=SAMPLE_RATE)
    if pcm.ndim != 2 or pcm.shape[-1] < 1:
        raise ValueError(f"invalid replay WAV: {path}")
    if pcm.shape[0] > 1:
        pcm = pcm.mean(axis=0, keepdims=True)
    return np.asarray(pcm[0], dtype=np.float32)


def replay_frames(
    paths: Sequence[Path],
    *,
    leading_silence_frames: int,
    between_silence_frames: int,
    tail_silence_frames: int,
) -> tuple[Iterator[Any], list[dict[str, object]]]:
    import numpy as np

    if not paths:
        raise ValueError("replay mode requires at least one --replay-wav")
    segments: list[tuple[str, Any]] = []
    boundaries: list[dict[str, object]] = []
    if leading_silence_frames:
        segments.append(("leading_silence", np.zeros(leading_silence_frames * FRAME_SIZE, np.float32)))
    cursor = leading_silence_frames
    for index, path in enumerate(paths):
        pcm = load_wav_24k(path)
        padding = (-pcm.size) % FRAME_SIZE
        if padding:
            pcm = np.pad(pcm, (0, padding))
        count = int(pcm.size // FRAME_SIZE)
        boundaries.append(
            {
                "path": str(path.resolve()),
                "start_frame": cursor,
                "end_frame": cursor + count - 1,
                "frames": count,
            }
        )
        segments.append((str(path), pcm))
        cursor += count
        if index + 1 < len(paths) and between_silence_frames:
            segments.append(("between_silence", np.zeros(between_silence_frames * FRAME_SIZE, np.float32)))
            cursor += between_silence_frames
    if tail_silence_frames:
        segments.append(("tail_silence", np.zeros(tail_silence_frames * FRAME_SIZE, np.float32)))

    def iterator() -> Iterator[Any]:
        for _, pcm in segments:
            for start in range(0, pcm.size, FRAME_SIZE):
                yield pcm[start : start + FRAME_SIZE]

    return iterator(), boundaries


def write_wav(path: Path, samples: Sequence[Any]) -> dict[str, object]:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    if samples:
        pcm = np.concatenate([np.asarray(item, dtype=np.float32).reshape(-1) for item in samples])
    else:
        pcm = np.zeros(0, dtype=np.float32)
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm16.tobytes())
    return {
        "path": str(path.resolve()),
        "samples": int(pcm.size),
        "duration_seconds": round(float(pcm.size / SAMPLE_RATE), 3),
        "rms": round(rms(pcm), 6),
        "peak": round(float(np.max(np.abs(pcm))) if pcm.size else 0.0, 6),
    }


class NativeLiveSession:
    """One persistent Mimi -> PersonaPlex/action/runtime -> Mimi state."""

    def __init__(self, args: argparse.Namespace) -> None:
        import sentencepiece
        import torch

        from moshi.models import loaders
        from moshi.models.agent_checkpoints import load_agent_lane_checkpoint
        from moshi.models.agent_loaders import get_personaplex_agent_micro_lm
        from moshi.models.agent_micro import AgentMicroLMGen
        from personaplex_agent.action_v2 import ActionKind, DEFAULT_MANIFEST
        from personaplex_agent.live_runtime_v2 import DuplexToolRuntimeV2
        from personaplex_agent.runtime_v2 import ActionV2RuntimeConstraint

        self.args = args
        self.torch = torch
        self.manifest = DEFAULT_MANIFEST
        self.ActionKind = ActionKind
        self.model_device = torch.device(args.model_device)
        self.mimi_device = torch.device(args.mimi_device)
        if self.model_device.type != "cuda" or self.mimi_device.type != "cuda":
            raise RuntimeError("the live 7B session requires CUDA devices")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        print(f"Loading streaming Mimi on {self.mimi_device}...", flush=True)
        with torch.cuda.device(self.mimi_device):
            self.mimi = loaders.get_mimi(args.mimi_weight, device=self.mimi_device)
        if int(self.mimi.sample_rate) != SAMPLE_RATE or float(self.mimi.frame_rate) != FRAME_RATE:
            raise RuntimeError("Mimi does not use the required 24 kHz / 12.5 Hz clock")

        print(f"Loading PersonaPlex and native action lanes on {self.model_device}...", flush=True)
        with torch.cuda.device(self.model_device):
            self.model = get_personaplex_agent_micro_lm(
                args.base_checkpoint,
                device=self.model_device,
                dtype=torch.bfloat16,
                action_card=256,
                environment_card=256,
                action_end_token_id=4,
                action_call_gate_threshold=args.call_gate_threshold,
            )
            self.lane_metadata = load_agent_lane_checkpoint(
                self.model,
                args.lane_checkpoint,
                expected_action_manifest_fingerprint=DEFAULT_MANIFEST.fingerprint,
            )
            self.model.eval()

        self.tokenizer = sentencepiece.SentencePieceProcessor(str(args.tokenizer))
        self.grammar = ActionV2RuntimeConstraint(1)
        self.runtime = DuplexToolRuntimeV2()
        self.vad = AcousticTurnWindow(
            threshold=args.vad_threshold,
            min_speech_frames=args.vad_min_speech_frames,
            end_silence_frames=args.vad_end_silence_frames,
            call_window_frames=args.call_window_frames,
        )
        self.executor = TimerExecutor()
        self.call_begin_id = DEFAULT_MANIFEST.action_id("CALL_BEGIN")
        self.noop_id = DEFAULT_MANIFEST.action_id("NOOP")
        self.call_count = 0
        self.event_seq = 0
        self.frame = 0
        self.current_frame_rms = 0.0
        self.current_vad_event: str | None = None
        self.frame_decision: dict[str, object] | None = None
        self.calls: list[dict[str, object]] = []
        self.terminal_results: list[dict[str, object]] = []
        self.pending_visible: list[tuple[int, str]] = []
        self.text_pieces: list[dict[str, object]] = []
        self.completions: list[dict[str, object]] = []
        self.frames: list[dict[str, object]] = []
        self.input_pcm: list[Any] = []
        self.output_pcm: list[Any] = []
        self.tool_completion_frame: int | None = None

        def mask_action_logits(
            slot: int, phase: Any, current_tokens: Any, logits: Any
        ) -> Any:
            masked = self.grammar.mask_logits(slot, phase, current_tokens, logits)
            if slot != 0:
                return masked
            masked = masked.clone()
            can_call = (
                self.call_count < args.max_calls
                and (not args.vad_gate_calls or self.vad.call_window_open)
            )
            if not can_call:
                masked[:, self.call_begin_id] = -float("inf")
            elif args.call_begin_logit_bias:
                masked[:, self.call_begin_id] += args.call_begin_logit_bias
            return masked

        def observe_action_frame(frames: Any, phases: Any, tokens: Any) -> None:
            self.grammar.commit_frame(frames, phases, tokens)

        def observe_action_decision(frames: Any, phases: Any, micro: Any) -> None:
            logits = micro.logits[0, 0].float()
            top_values, top_ids = logits.topk(5)
            call_value = float(logits[self.call_begin_id])
            noop_value = float(logits[self.noop_id])
            self.frame_decision = {
                "frame": int(frames[0]),
                "phase": int(phases[0]),
                "combined_gate_logit": float(micro.call_gate_logits[0]),
                "endpoint_logit": float(micro.call_endpoint_logits[0]),
                "intent_logit": float(micro.call_intent_logits[0]),
                "threshold": float(self.model.action_call_gate_threshold),
                "call_window_open": self.vad.call_window_open,
                "call_begin_logit": call_value,
                "noop_logit": noop_value,
                "call_begin_minus_noop": call_value - noop_value,
                "top_start_ids": [int(value) for value in top_ids.tolist()],
                "top_start_tokens": [
                    DEFAULT_MANIFEST.action_name(int(value)) for value in top_ids.tolist()
                ],
                "top_start_logits": [float(value) for value in top_values.tolist()],
            }

        def observe_completion(frames: Any, eos_started: Any, sentence_started: Any) -> None:
            for index in range(frames.shape[0]):
                if bool(eos_started[index]):
                    self.completions.append({"frame": int(frames[index]), "kind": "text_eos"})
                if bool(sentence_started[index]):
                    value = {"frame": int(frames[index]), "kind": "sentence_end"}
                    self.completions.append(value)
                    if self.call_count:
                        self.tool_completion_frame = int(frames[index])

        sentence_ids = tuple(
            dict.fromkeys(
                token
                for token in (
                    int(self.tokenizer.piece_to_id(".")),
                    int(self.tokenizer.piece_to_id("!")),
                    int(self.tokenizer.piece_to_id("?")),
                )
                if token >= 4
            )
        )
        self.generator = AgentMicroLMGen(
            self.model,
            audio_tokens_per_stream=8,
            use_sampling=True,
            use_action_sampling=False,
            temp=args.audio_temperature,
            temp_text=args.text_temperature,
            temp_action=args.action_temperature,
            top_k=args.audio_top_k,
            top_k_text=args.text_top_k,
            top_k_action=args.action_top_k,
            action_logit_masker=mask_action_logits,
            action_frame_observer=observe_action_frame,
            action_decision_observer=observe_action_decision,
            mute_speech_while_tool_pending=True,
            pending_speech_call_begin_token_id=DEFAULT_MANIFEST.action_id("CALL_BEGIN"),
            pending_speech_environment_end_token_id=DEFAULT_MANIFEST.environment_id("OBS_END"),
            pending_speech_terminal_status_ids=tuple(
                DEFAULT_MANIFEST.environment_id(token)
                for token in ("OK", "ERROR", "TIMEOUT", "CANCELLED", "STALE")
            ),
            stop_speech_on_result_sentence_end=True,
            text_sentence_end_token_ids=sentence_ids,
            result_sentence_min_semantic_tokens=2,
            greedy_first_result_semantic_token=True,
            turn_completion_observer=observe_completion,
            cuda_graph_backbone=args.cuda_graph_backbone,
            cuda_graph_audio=args.cuda_graph_audio,
        )
        self.generator.load_voice_prompt_embeddings(args.voice_prompt)
        self.generator.text_prompt_tokens = tuple(
            self.tokenizer.encode(wrap_with_system_tags(args.text_prompt))
        )
        self.generator.audio_silence_frame_cnt = args.audio_prompt_silence_frames
        self._mimi_context: Any = None
        self._generator_context: Any = None

    def __enter__(self) -> "NativeLiveSession":
        torch = self.torch
        self._mimi_context = self.mimi.streaming(1)
        self._generator_context = self.generator.streaming(1)
        self._mimi_context.__enter__()
        self._generator_context.__enter__()
        with torch.inference_mode(), torch.cuda.device(self.model_device):
            origin = self.generator.step_system_prompts()
        if origin != 0 or self.generator.next_action_frame != 0:
            raise AssertionError("live generator prompt did not reset to frame zero")
        self.mimi.reset_streaming()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._generator_context is not None:
            self._generator_context.__exit__(exc_type, exc, traceback)
        if self._mimi_context is not None:
            self._mimi_context.__exit__(exc_type, exc, traceback)

    def _schedule_result(self, action: Any, execution: ToolExecution, lease: Any) -> None:
        from personaplex_agent.environment_v2 import EnvironmentEventV2, EnvironmentKind

        pending_id = f"live-call-{self.call_count}-pending"
        terminal_id = f"live-call-{self.call_count}-terminal"
        self.runtime.submit_event(
            EnvironmentEventV2(
                event_id=pending_id,
                event_seq=self.event_seq,
                available_frame=lease.issued_frame + 1,
                issued_frame=lease.issued_frame,
                ref=lease.ref,
                tool_name=action.tool_name,
                kind=EnvironmentKind.PENDING,
                call_uuid=lease.call_uuid,
                lease=lease.lease,
            )
        )
        self.event_seq += 1
        self.runtime.submit_event(
            EnvironmentEventV2(
                event_id=terminal_id,
                event_seq=self.event_seq,
                available_frame=lease.issued_frame + self.args.result_latency_frames,
                issued_frame=lease.issued_frame,
                ref=lease.ref,
                tool_name=action.tool_name,
                kind=EnvironmentKind.OK if execution.ok else EnvironmentKind.ERROR,
                payload_tokens=execution.payload_tokens,
                call_uuid=lease.call_uuid,
                lease=lease.lease,
            )
        )
        self.event_seq += 1

    def process(self, pcm: Any) -> Any:
        import numpy as np

        torch = self.torch
        values = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if values.size != FRAME_SIZE:
            raise ValueError(f"live PCM frame must contain {FRAME_SIZE} samples")
        self.input_pcm.append(values.copy())
        self.current_frame_rms = rms(values)
        self.current_vad_event = self.vad.update(self.current_frame_rms)
        self.frame_decision = None

        for visible_frame, text in tuple(self.pending_visible):
            if visible_frame == self.frame:
                print(f"\nTOOL RESULT VISIBLE TO MODEL: {text}", flush=True)
                self.pending_visible.remove((visible_frame, text))

        environment = self.runtime.environment_frame(self.frame)
        for completed in environment.completed_events:
            if completed.event.is_terminal:
                item = {
                    "event_id": completed.event.event_id,
                    "ref": completed.event.ref,
                    "kind": completed.event.kind.value,
                    "payload_tokens": list(completed.event.payload_tokens),
                    "injected_end_frame": completed.injected_end.frame,
                    "visible_frame": completed.visible_frame,
                }
                self.terminal_results.append(item)
                human = self.calls[-1]["result_text"] if self.calls else completed.event.kind.value
                self.pending_visible.append((completed.visible_frame, str(human)))

        component_timings: dict[str, float] | None = None
        if self.args.profile_components:
            component_timings = {}
            torch.cuda.synchronize(self.mimi_device)
            torch.cuda.synchronize(self.model_device)
            component_started = time.perf_counter()

        with torch.inference_mode():
            with torch.cuda.device(self.mimi_device):
                waveform = torch.as_tensor(values, device=self.mimi_device).view(1, 1, -1)
                codes = self.mimi.encode(waveform)
            if component_timings is not None:
                torch.cuda.synchronize(self.mimi_device)
                now = time.perf_counter()
                component_timings["mimi_encode_ms"] = round(
                    (now - component_started) * 1000, 3
                )
                component_started = now
            if codes.shape != (1, 8, 1):
                raise AssertionError(f"expected one Mimi frame, got {tuple(codes.shape)}")
            environment_tensor = torch.tensor(
                [environment.ids], dtype=torch.long, device=self.model_device
            )
            with torch.cuda.device(self.model_device):
                generated = self.generator.step(
                    codes.to(device=self.model_device, dtype=torch.long),
                    environment_tokens=environment_tensor,
                )
            if component_timings is not None:
                torch.cuda.synchronize(self.model_device)
                now = time.perf_counter()
                component_timings["model_step_ms"] = round(
                    (now - component_started) * 1000, 3
                )
                component_started = now

        completed_actions: list[dict[str, object]] = []
        for batch_index, action in self.grammar.drain_completed():
            if batch_index != 0:
                raise AssertionError("live session unexpectedly emitted another batch")
            if action.kind is not self.ActionKind.CALL:
                continue
            dispatch = self.runtime.dispatch_action(action)
            assert dispatch.lease is not None and action.tool_name is not None
            arguments = dict(action.arguments)
            execution = self.executor.execute(action.tool_name, arguments)
            self.call_count += 1
            call = {
                "index": self.call_count - 1,
                "tool_name": action.tool_name,
                "ref": action.ref,
                "arguments": arguments,
                "issued_frame": action.issued_frame,
                "result_ok": execution.ok,
                "result_tokens": list(execution.payload_tokens),
                "result_text": execution.human_text,
                "scheduled_seconds": execution.scheduled_seconds,
            }
            self.calls.append(call)
            completed_actions.append(call)
            print(f"\nTOOL CALL: {action.tool_name} {json.dumps(arguments)}", flush=True)
            print(f"TOOL EXECUTOR: {execution.human_text}", flush=True)
            self._schedule_result(action, execution, dispatch.lease)
            self.grammar.set_active_refs(0, self.runtime.active_refs)

        released = self.runtime.acknowledge_model_frame(self.frame)
        if released:
            self.grammar.set_active_refs(0, self.runtime.active_refs)

        output = np.zeros(FRAME_SIZE, dtype=np.float32)
        text_piece = None
        text_token = None
        if generated is not None:
            text_token = int(generated.text_token[0].item())
            if text_token >= 4:
                text_piece = str(self.tokenizer.id_to_piece(text_token)).replace("▁", " ")
                self.text_pieces.append(
                    {"frame": self.frame, "token": text_token, "piece": text_piece}
                )
                print(text_piece, end="", flush=True)
            with torch.inference_mode(), torch.cuda.device(self.mimi_device):
                audio_tokens = generated.assistant_audio_tokens.to(
                    device=self.mimi_device, dtype=torch.long
                ).unsqueeze(-1)
                decoded = self.mimi.decode(audio_tokens)
            if component_timings is not None:
                torch.cuda.synchronize(self.mimi_device)
                now = time.perf_counter()
                component_timings["mimi_decode_ms"] = round(
                    (now - component_started) * 1000, 3
                )
            output = decoded[0, 0].detach().to(device="cpu").float().numpy()
            if output.size != FRAME_SIZE:
                raise AssertionError(f"Mimi decoded {output.size} samples, expected {FRAME_SIZE}")
        self.output_pcm.append(output.copy())

        row = {
            "frame": self.frame,
            "input_rms": self.current_frame_rms,
            "vad_event": self.current_vad_event,
            "vad_in_speech": self.vad.in_speech,
            "call_window_remaining": self.vad.call_window_remaining,
            "text_token": text_token,
            "text_piece": text_piece,
            "completed_actions": completed_actions,
            "released_refs": [item.ref for item in released],
            "decision": self.frame_decision,
            "component_timings": component_timings,
        }
        self.frames.append(row)
        self.vad.advance()
        self.frame += 1
        return output

    def should_auto_stop(self) -> bool:
        return bool(
            self.args.auto_stop_after_tool_response
            and self.tool_completion_frame is not None
            and self.frame >= self.tool_completion_frame + self.args.flush_frames
        )

    def report(
        self,
        *,
        mode: str,
        elapsed_seconds: float,
        replay_boundaries: Sequence[dict[str, object]],
        audio_io: LiveSoundDevice | None,
    ) -> dict[str, object]:
        text = "".join(str(item["piece"]) for item in self.text_pieces).strip()
        return {
            "live_session_v2": "ok",
            "runner_version": RUNNER_VERSION,
            "mode": mode,
            "persistent_model_state": True,
            "streaming_mimi_encode": True,
            "streaming_mimi_decode": True,
            "external_asr": False,
            "external_planner": False,
            "model_device": str(self.model_device),
            "mimi_device": str(self.mimi_device),
            "lane_checkpoint": str(self.args.lane_checkpoint.resolve()),
            "lane_checkpoint_sha256": sha256(self.args.lane_checkpoint),
            "lane_metadata": self.lane_metadata,
            "seed": self.args.seed,
            "frames": self.frame,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "realtime_factor": round(elapsed_seconds / max(self.frame / FRAME_RATE, 1e-9), 3),
            "vad": {
                "threshold": self.vad.threshold,
                "min_speech_frames": self.vad.min_speech_frames,
                "end_silence_frames": self.vad.end_silence_frames,
                "call_window_frames": self.vad.call_window_frames,
                "utterance_count": self.vad.utterance_count,
            },
            "cuda_graphs": {
                "backbone": self.args.cuda_graph_backbone,
                "audio": self.args.cuda_graph_audio,
            },
            "replay_boundaries": list(replay_boundaries),
            "calls": self.calls,
            "terminal_results_visible_to_model": self.terminal_results,
            "generated_assistant_text": text,
            "turn_completion_events": self.completions,
            "audio_io": None
            if audio_io is None
            else {
                "input_statuses": audio_io.input_statuses,
                "output_statuses": audio_io.output_statuses,
                "dropped_input_frames": audio_io.dropped_input_frames,
                "buffered_input_frames_at_stop": audio_io.buffered_input_frames,
            },
            "frame_trace": self.frames,
        }


def parse_args() -> argparse.Namespace:
    model_root = ROOT / "models" / "personaplex-7b-v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("microphone", "replay"), default="microphone")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--input-device")
    parser.add_argument("--output-device")
    parser.add_argument("--no-playback", action="store_true")
    parser.add_argument("--max-session-seconds", type=float, default=180.0)
    parser.add_argument("--replay-wav", action="append", type=Path, default=[])
    parser.add_argument("--leading-silence-frames", type=int, default=0)
    parser.add_argument("--between-silence-frames", type=int, default=50)
    parser.add_argument("--tail-silence-frames", type=int, default=96)
    parser.add_argument("--base-checkpoint", type=Path, default=model_root / "model.safetensors")
    parser.add_argument(
        "--lane-checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "micro-head-v169-v151-dual-state-margin05.safetensors",
    )
    parser.add_argument("--tokenizer", type=Path, default=model_root / "tokenizer_spm_32k_3.model")
    parser.add_argument(
        "--mimi-weight",
        type=Path,
        default=model_root / "tokenizer-e351c8d8-checkpoint125.safetensors",
    )
    parser.add_argument(
        "--voice-prompt",
        type=Path,
        default=model_root / "voices" / "voices" / "NATM1.pt",
    )
    parser.add_argument(
        "--text-prompt",
        default="You are a concise, helpful voice assistant. Respond naturally and accurately to the user.",
    )
    parser.add_argument("--audio-prompt-silence-frames", type=int, default=6)
    parser.add_argument("--mimi-device", default="cuda:1")
    parser.add_argument("--model-device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--text-temperature", type=float, default=0.7)
    parser.add_argument("--action-temperature", type=float, default=0.7)
    parser.add_argument("--audio-temperature", type=float, default=0.8)
    parser.add_argument("--text-top-k", type=int, default=1)
    parser.add_argument("--action-top-k", type=int, default=25)
    parser.add_argument("--audio-top-k", type=int, default=250)
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument(
        "--cuda-graph-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--cuda-graph-audio",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--call-gate-threshold", type=float, default=0.0)
    parser.add_argument("--call-begin-logit-bias", type=float, default=0.125)
    parser.add_argument("--max-calls", type=int, default=1)
    parser.add_argument("--result-latency-frames", type=int, default=3)
    parser.add_argument("--vad-threshold", type=float, default=0.008)
    parser.add_argument("--vad-min-speech-frames", type=int, default=2)
    parser.add_argument("--vad-end-silence-frames", type=int, default=2)
    parser.add_argument("--call-window-frames", type=int, default=24)
    parser.add_argument(
        "--vad-gate-calls",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--auto-stop-after-tool-response",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--flush-frames", type=int, default=16)
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=ROOT / "artifacts" / "live-session-v2" / "last-session.json",
    )
    parser.add_argument(
        "--input-audio-output",
        type=Path,
        default=ROOT / "artifacts" / "live-session-v2" / "last-input.wav",
    )
    parser.add_argument(
        "--assistant-audio-output",
        type=Path,
        default=ROOT / "artifacts" / "live-session-v2" / "last-assistant.wav",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.mimi_device.casefold() == "cuda:2" or args.model_device.casefold() == "cuda:2":
        raise ValueError("GPU 2 is excluded from this project run by user request")
    for path in (
        args.base_checkpoint,
        args.lane_checkpoint,
        args.tokenizer,
        args.mimi_weight,
        args.voice_prompt,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(args.lane_checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("live-session lane checkpoint SHA-256 mismatch")
    if args.mode == "replay":
        if not args.replay_wav:
            raise ValueError("replay mode requires --replay-wav")
        for path in args.replay_wav:
            if not path.is_file():
                raise FileNotFoundError(path)
    for name in (
        "leading_silence_frames",
        "between_silence_frames",
        "tail_silence_frames",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} cannot be negative")
    if args.max_session_seconds <= 0:
        raise ValueError("max-session-seconds must be positive")
    if args.max_calls < 1:
        raise ValueError("max-calls must be positive")
    if args.result_latency_frames < 1 or args.flush_frames < 1:
        raise ValueError("result latency and flush frames must be positive")


def main() -> int:
    args = parse_args()
    if args.list_devices:
        return list_audio_devices()
    validate_args(args)

    import numpy as np
    import torch

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    audio_io: LiveSoundDevice | None = None
    boundaries: list[dict[str, object]] = []
    if args.mode == "replay":
        source, boundaries = replay_frames(
            args.replay_wav,
            leading_silence_frames=args.leading_silence_frames,
            between_silence_frames=args.between_silence_frames,
            tail_silence_frames=args.tail_silence_frames,
        )
    else:
        source = iter(())

    load_started = time.perf_counter()
    try:
        with NativeLiveSession(args) as session:
            model_load_seconds = time.perf_counter() - load_started
            session_started = time.perf_counter()
            print("\n=== PERSISTENT PERSONAPLEX SESSION LIVE ===", flush=True)
            if args.mode == "microphone":
                print("Use headphones. Speak naturally; press Ctrl+C to stop.", flush=True)
                print("Later say: Count down 2 minutes for me.", flush=True)
                audio_io = LiveSoundDevice(
                    input_device=_coerce_device(args.input_device),
                    output_device=_coerce_device(args.output_device),
                    playback=not args.no_playback,
                )
                audio_io.__enter__()

            deadline = session_started + args.max_session_seconds
            while args.mode == "replay" or time.perf_counter() < deadline:
                if args.mode == "microphone":
                    assert audio_io is not None
                    block = audio_io.read()
                    if block is None:
                        continue
                else:
                    try:
                        block = next(source)
                    except StopIteration:
                        break
                frame_started = time.perf_counter()
                output = session.process(block)
                processing_ms = (time.perf_counter() - frame_started) * 1000
                session.frames[-1]["processing_ms"] = round(processing_ms, 3)
                if audio_io is not None:
                    audio_io.play(output)
                    backlog = audio_io.buffered_input_frames
                    if backlog and session.frame % 25 == 0:
                        print(
                            f"\n[latency] microphone backlog: {backlog} x 80 ms frames",
                            flush=True,
                        )
                if session.should_auto_stop():
                    print("\nTool response completed; flushing the live session.", flush=True)
                    break

            if audio_io is not None:
                audio_io.drain_playback()
            elapsed = time.perf_counter() - session_started
            report = session.report(
                mode=args.mode,
                elapsed_seconds=elapsed,
                replay_boundaries=boundaries,
                audio_io=audio_io,
            )
            report["model_load_seconds"] = round(model_load_seconds, 3)
            report["input_audio"] = write_wav(args.input_audio_output, session.input_pcm)
            report["assistant_audio"] = write_wav(
                args.assistant_audio_output, session.output_pcm
            )
    except KeyboardInterrupt:
        print("\nStopping live session...", flush=True)
        if "session" not in locals():
            return 130
        elapsed = time.perf_counter() - session_started
        report = session.report(
            mode=args.mode,
            elapsed_seconds=elapsed,
            replay_boundaries=boundaries,
            audio_io=audio_io,
        )
        report["model_load_seconds"] = round(model_load_seconds, 3)
        report["input_audio"] = write_wav(args.input_audio_output, session.input_pcm)
        report["assistant_audio"] = write_wav(
            args.assistant_audio_output, session.output_pcm
        )
    finally:
        if audio_io is not None:
            audio_io.__exit__(None, None, None)

    report["trace_output"] = str(args.trace_output.resolve())
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {
        "live_session_v2": report["live_session_v2"],
        "mode": report["mode"],
        "frames": report["frames"],
        "model_load_seconds": report["model_load_seconds"],
        "elapsed_seconds": report["elapsed_seconds"],
        "realtime_factor": report["realtime_factor"],
        "calls": report["calls"],
        "terminal_results_visible_to_model": report[
            "terminal_results_visible_to_model"
        ],
        "generated_assistant_text": report["generated_assistant_text"],
        "input_audio": report["input_audio"],
        "assistant_audio": report["assistant_audio"],
        "trace_output": report["trace_output"],
    }
    print("\n=== SESSION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

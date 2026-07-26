"""Causal training-episode contract for native tool-aware duplex training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .native_lane import ACTION_PACKET_FRAMES, ActionPacket, EnvironmentPacket
from .protocol import ActionCall, JSONValue, ObservationKind, ToolStatus


@dataclass(frozen=True, slots=True)
class EpisodeObservation:
    frame: int
    call_id: str
    kind: ObservationKind = ObservationKind.RESULT
    status: ToolStatus | None = None
    payload: dict[str, JSONValue] | None = None

    def __post_init__(self) -> None:
        if self.frame < 0:
            raise ValueError("observation frame must be non-negative")
        if self.kind is ObservationKind.PENDING:
            if self.status is not None or self.payload is not None:
                raise ValueError("pending observations cannot carry a terminal status or payload")
        elif self.status is None or self.payload is None:
            raise ValueError("result observations require a status and payload")


@dataclass(frozen=True, slots=True)
class DuplexToolEpisode:
    """A single aligned duplex/audio/action/result training episode."""

    episode_id: str
    sample_rate: int
    frame_rate_hz: float
    user_audio: str
    assistant_audio: str
    calls: tuple[ActionCall, ...]
    action_packets: tuple[ActionPacket, ...]
    observations: tuple[EpisodeObservation, ...]
    environment_packets: tuple[EnvironmentPacket, ...]

    def validate(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id is required")
        if self.sample_rate != 24000:
            raise ValueError("PersonaPlex-Agent training episodes must use 24 kHz audio")
        if self.frame_rate_hz != 12.5:
            raise ValueError("PersonaPlex-Agent training episodes must use a 12.5 Hz / 80 ms clock")
        if not self.user_audio or not self.assistant_audio:
            raise ValueError("both user and assistant audio references are required")

        calls = {call.call_id: call for call in self.calls}
        if len(calls) != len(self.calls):
            raise ValueError("call IDs must be unique within an episode")

        for packet in self.action_packets:
            if packet.start_frame % ACTION_PACKET_FRAMES:
                raise ValueError("action packet is not 160 ms aligned")
            if packet.call_id is not None and packet.call_id not in calls:
                raise ValueError(f"action packet references unknown call {packet.call_id}")

        # An executor may only see a call after the final token that made it
        # grammar-valid.  ``issued_frame`` is therefore a dispatch/CALL_END
        # frame, not merely the start of the first action packet.
        final_action_frame: dict[str, int] = {}
        for packet in self.action_packets:
            if packet.call_id is None:
                continue
            packet_final = max(
                frame
                for frame in (packet.start_frame, packet.start_frame + 1)
                if packet.tokens_for_frame(frame)
            )
            final_action_frame[packet.call_id] = max(
                final_action_frame.get(packet.call_id, -1), packet_final
            )
        for call_id, call in calls.items():
            final_frame = final_action_frame.get(call_id)
            if final_frame is None:
                raise ValueError(f"call {call_id} has no generated action packet")
            if call.issued_frame != final_frame:
                raise ValueError(
                    f"call {call_id} issued_frame must equal final action frame {final_frame}"
                )

        observation_keys: set[tuple[str, int, ObservationKind]] = set()
        for observation in self.observations:
            call = calls.get(observation.call_id)
            if call is None:
                raise ValueError(f"observation references unknown call {observation.call_id}")
            if observation.frame <= call.issued_frame:
                raise ValueError("tool observations must occur strictly after their action")
            key = (observation.call_id, observation.frame, observation.kind)
            if key in observation_keys:
                raise ValueError("duplicate typed observation at the same causal frame")
            observation_keys.add(key)

        for packet in self.environment_packets:
            call = calls.get(packet.call_id)
            if call is None:
                raise ValueError(f"environment packet references unknown call {packet.call_id}")
            if packet.frame <= call.issued_frame:
                raise ValueError("environment packet leaks a result at or before its action")
            if (packet.call_id, packet.frame, packet.kind) not in observation_keys:
                raise ValueError("environment packet requires a matching typed observation")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DuplexToolEpisode":
        def as_tokens(raw: list[Any]) -> tuple[str, ...]:
            if not all(isinstance(token, str) for token in raw):
                raise ValueError("packet tokens must be strings before vocabulary encoding")
            return tuple(raw)

        calls = tuple(
            ActionCall(
                call_id=item["call_id"],
                tool_name=item["tool_name"],
                arguments=item["arguments"],
                issued_frame=item["issued_frame"],
            )
            for item in data["calls"]
        )
        actions = tuple(
            ActionPacket(
                start_frame=item["start_frame"],
                call_id=item.get("call_id"),
                tokens=as_tokens(item["tokens"]),
            )
            for item in data["action_packets"]
        )
        observations = tuple(
            EpisodeObservation(
                frame=item["frame"],
                call_id=item["call_id"],
                kind=ObservationKind(item.get("kind", "result")),
                status=(ToolStatus(item["status"]) if item.get("status") is not None else None),
                payload=item.get("payload"),
            )
            for item in data["observations"]
        )
        environment = tuple(
            EnvironmentPacket(
                frame=item["frame"],
                call_id=item["call_id"],
                tokens=as_tokens(item["tokens"]),
                kind=ObservationKind(item.get("kind", "result")),
                source=item.get("source", "tool_executor"),
            )
            for item in data["environment_packets"]
        )
        episode = cls(
            episode_id=data["episode_id"],
            sample_rate=data["sample_rate"],
            frame_rate_hz=data["frame_rate_hz"],
            user_audio=data["user_audio"],
            assistant_audio=data["assistant_audio"],
            calls=calls,
            action_packets=actions,
            observations=observations,
            environment_packets=environment,
        )
        episode.validate()
        return episode


def load_episode(path: str | Path) -> DuplexToolEpisode:
    """Load and causally validate a JSON training episode."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DuplexToolEpisode.from_dict(data)

"""Native action/result-state primitives for PersonaPlex-Agent."""

from .action_v2 import (
    DEFAULT_MANIFEST,
    ActionCompiler,
    ActionKind,
    ActionPacketV2,
    ActionParser,
    ActionV2Manifest,
    ControlAction,
    MicroLane,
    ParsedAction,
    RefLeaseGuard,
    materialize_action_lane,
)
from .base_codes_v2 import (
    PersonaPlexBaseCodes,
    assemble_personaplex_base_codes,
)
from .bridge import CausalToolBridge, ToolRegistry
from .corpus_v2 import (
    CorpusReplayVerifier,
    FiniteDomainSyntheticExecutor,
    ScenarioSpec,
    SymbolicCorpusBuilder,
    SymbolicCorpusRecord,
)
from .environment_v2 import (
    EnvironmentEventV2,
    EnvironmentFifoScheduler,
    EnvironmentKind,
    EnvironmentMicroLane,
    EnvironmentSchedule,
)
from .evaluation_v2 import (
    CaseEvaluationV2,
    CaseMetricsV2,
    EvaluationReportV2,
    ExpectedCallV2,
    ExpectedGroundingV2,
    GroundedClaimTraceV2,
    HeldOutCaseV2,
    HeldOutV2Evaluator,
    InferenceTraceV2,
    TerminalEventTraceV2,
)
from .live_runtime_v2 import (
    ActionDispatchV2,
    DuplexToolRuntimeV2,
    LiveEnvironmentFifo,
    LiveEnvironmentFrame,
)
from .protocol import ActionCall, ObservationEvent, ObservationKind, ToolOutcome, ToolStatus
from .scenario_catalog_v2 import (
    ScenarioSplit,
    SpeechScenarioFactoryV2,
    SpeechScenarioManifestV2,
    SpeechScenarioV2,
    build_default_speech_scenario_manifest,
)
from .text_alignment_v2 import AlignedTextTokensV2, uniformly_align_text_tokens
from .training_masks_v2 import (
    ActionSupervisionMaskV2,
    build_balanced_action_supervision,
)
from .training import (
    MicroFrameTargets,
    TypedLaneVocabulary,
    V1FrameTargets,
    build_micro_frame_targets,
    build_v1_frame_targets,
)

__all__ = [
    "ActionCall",
    "ActionCompiler",
    "ActionDispatchV2",
    "ActionKind",
    "ActionPacketV2",
    "ActionParser",
    "ActionV2Manifest",
    "AlignedTextTokensV2",
    "CausalToolBridge",
    "CaseEvaluationV2",
    "CaseMetricsV2",
    "ControlAction",
    "CorpusReplayVerifier",
    "DEFAULT_MANIFEST",
    "EnvironmentEventV2",
    "EnvironmentFifoScheduler",
    "EnvironmentKind",
    "EnvironmentMicroLane",
    "EnvironmentSchedule",
    "EvaluationReportV2",
    "ExpectedCallV2",
    "ExpectedGroundingV2",
    "FiniteDomainSyntheticExecutor",
    "GroundedClaimTraceV2",
    "HeldOutCaseV2",
    "HeldOutV2Evaluator",
    "InferenceTraceV2",
    "MicroLane",
    "DuplexToolRuntimeV2",
    "LiveEnvironmentFifo",
    "LiveEnvironmentFrame",
    "ObservationEvent",
    "ObservationKind",
    "ParsedAction",
    "PersonaPlexBaseCodes",
    "RefLeaseGuard",
    "ScenarioSpec",
    "ScenarioSplit",
    "SpeechScenarioFactoryV2",
    "SpeechScenarioManifestV2",
    "SpeechScenarioV2",
    "SymbolicCorpusBuilder",
    "SymbolicCorpusRecord",
    "ToolOutcome",
    "ToolRegistry",
    "ToolStatus",
    "TerminalEventTraceV2",
    "ActionSupervisionMaskV2",
    "MicroFrameTargets",
    "TypedLaneVocabulary",
    "V1FrameTargets",
    "build_micro_frame_targets",
    "build_v1_frame_targets",
    "assemble_personaplex_base_codes",
    "build_balanced_action_supervision",
    "build_default_speech_scenario_manifest",
    "materialize_action_lane",
    "uniformly_align_text_tokens",
]

"""Attack implementations for DoSRAGBench prototype."""

from dosragbench.attacks.a1_guardrail import GuardrailTriggeringAttack
from dosragbench.attacks.a2_contradiction import ContradictionFloodingAttack
from dosragbench.attacks.a3_authority import AuthoritySpoofingAttack
from dosragbench.attacks.base import DoSAttack
from dosragbench.attacks.c0_random_baseline import RandomInjectionAttack
from dosragbench.attacks.c1_clustering import EmbeddingClusteringAttack
from dosragbench.attacks.c2_index_pollution import IndexPollutionAttack
from dosragbench.attacks.c3_embedding_perturbation import EmbeddingPerturbationAttack
from dosragbench.attacks.b1_context_saturation import ContextSaturationAttack
from dosragbench.attacks.b2_generation_loop import GenerationLoopAttack
from dosragbench.attacks.b3_multi_retrieval import MultiRetrievalAmplificationAttack
from dosragbench.attacks.d1_logical_contradiction import LogicalContradictionAttack
from dosragbench.attacks.d2_circular_reference import CircularReferenceAttack
from dosragbench.attacks.d3_epistemic_uncertainty import EpistemicUncertaintyAttack
from dosragbench.attacks.d4_infinite_qualification import InfiniteQualificationAttack
from dosragbench.utils.config import AttackConfig


ATTACK_REGISTRY: dict[str, type[DoSAttack]] = {
    "A1": GuardrailTriggeringAttack,
    "A2": ContradictionFloodingAttack,
    "A3": AuthoritySpoofingAttack,
    "RAND": RandomInjectionAttack,
    "B1": ContextSaturationAttack,
    "B2": GenerationLoopAttack,
    "B3": MultiRetrievalAmplificationAttack,
    "C1": EmbeddingClusteringAttack,
    "C2": IndexPollutionAttack,
    "C3": EmbeddingPerturbationAttack,
    "D1": LogicalContradictionAttack,
    "D2": CircularReferenceAttack,
    "D3": EpistemicUncertaintyAttack,
    "D4": InfiniteQualificationAttack,
}


def build_attack(config: AttackConfig, **kwargs) -> DoSAttack:
    """Factory for constructing an attack from its config."""
    if config.category not in ATTACK_REGISTRY:
        available = ", ".join(ATTACK_REGISTRY.keys())
        raise KeyError(
            f"Attack category '{config.category}' not implemented. Available: {available}"
        )
    attack_cls = ATTACK_REGISTRY[config.category]
    return attack_cls(config, **kwargs) if kwargs else attack_cls(config)


__all__ = [
    "DoSAttack",
    "GuardrailTriggeringAttack",
    "ContradictionFloodingAttack",
    "AuthoritySpoofingAttack",
    "RandomInjectionAttack",
    "EmbeddingClusteringAttack",
    "IndexPollutionAttack",
    "EmbeddingPerturbationAttack",
    "ContextSaturationAttack",
    "GenerationLoopAttack",
    "MultiRetrievalAmplificationAttack",
    "LogicalContradictionAttack",
    "CircularReferenceAttack",
    "EpistemicUncertaintyAttack",
    "InfiniteQualificationAttack",
    "ATTACK_REGISTRY",
    "build_attack",
]

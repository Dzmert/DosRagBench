"""Attack implementations for DoSRAGBench prototype."""

from dosragbench.attacks.a1_guardrail import GuardrailTriggeringAttack
from dosragbench.attacks.a2_contradiction import ContradictionFloodingAttack
from dosragbench.attacks.a3_authority import AuthoritySpoofingAttack
from dosragbench.attacks.base import DoSAttack
from dosragbench.attacks.c0_random_baseline import RandomInjectionAttack
from dosragbench.attacks.c1_clustering import EmbeddingClusteringAttack
from dosragbench.attacks.c2_index_pollution import IndexPollutionAttack
from dosragbench.attacks.c3_embedding_perturbation import EmbeddingPerturbationAttack
from dosragbench.utils.config import AttackConfig


ATTACK_REGISTRY: dict[str, type[DoSAttack]] = {
    "A1": GuardrailTriggeringAttack,
    "A2": ContradictionFloodingAttack,
    "A3": AuthoritySpoofingAttack,
    "RAND": RandomInjectionAttack,
    "C1": EmbeddingClusteringAttack,
    "C2": IndexPollutionAttack,
    "C3": EmbeddingPerturbationAttack,
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
    "ATTACK_REGISTRY",
    "build_attack",
]

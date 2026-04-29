"""Attack implementations for DoSRAGBench prototype."""

from dosragbench.attacks.a1_guardrail import GuardrailTriggeringAttack
from dosragbench.attacks.base import DoSAttack
from dosragbench.attacks.c1_clustering import EmbeddingClusteringAttack
from dosragbench.utils.config import AttackConfig


ATTACK_REGISTRY: dict[str, type[DoSAttack]] = {
    "A1": GuardrailTriggeringAttack,
    "A1_instructional": GuardrailTriggeringAttack,
    "A1_reframe": GuardrailTriggeringAttack,
    "C1": EmbeddingClusteringAttack,
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
    "EmbeddingClusteringAttack",
    "ATTACK_REGISTRY",
    "build_attack",
]

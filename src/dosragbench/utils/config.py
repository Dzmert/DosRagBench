"""Configuration and path utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Project paths
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
CONFIGS_DIR = REPO_ROOT / "configs"
CACHE_DIR = Path(os.environ.get("DOSRAGBENCH_CACHE", REPO_ROOT / ".cache"))

for d in (DATA_DIR, RESULTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Specification for a single model."""

    name: str
    hf_id: str
    alignment_level: int  # 0 = base, 1 = light, 2 = moderate, 3 = heavy, 4 = reasoning
    quantization: str = "4bit"  # "4bit", "8bit", "none"
    chat_template: bool = True  # False for base models
    max_new_tokens: int = 256


@dataclass
class ModelPairConfig:
    """Matched base/instruct pair for alignment-effect isolation."""

    pair_name: str
    base: ModelConfig
    aligned: ModelConfig


@dataclass
class AttackConfig:
    """Attack specification."""

    category: str  # "A1", "C1", etc.
    name: str
    num_queries: int = 50
    num_adversarial_docs: int = 5
    seed: int = 42
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    """Full experimental run configuration."""

    model_pair: ModelPairConfig
    attack: AttackConfig
    embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    kb_size: int = 1000
    top_k: int = 5
    device: str = "auto"


def load_yaml(path: Path) -> dict:
    """Load a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model_pair(pair_name: str) -> ModelPairConfig:
    """Load a matched model pair config by name."""
    cfg = load_yaml(CONFIGS_DIR / "model_pairs.yaml")
    if pair_name not in cfg:
        available = ", ".join(cfg.keys())
        raise KeyError(f"Unknown model pair '{pair_name}'. Available: {available}")

    pair = cfg[pair_name]
    return ModelPairConfig(
        pair_name=pair_name,
        base=ModelConfig(**pair["base"]),
        aligned=ModelConfig(**pair["aligned"]),
    )


def load_attack(category: str) -> AttackConfig:
    """Load an attack config by category code."""
    cfg = load_yaml(CONFIGS_DIR / "attacks.yaml")
    if category not in cfg:
        available = ", ".join(cfg.keys())
        raise KeyError(f"Unknown attack '{category}'. Available: {available}")
    return AttackConfig(category=category, **cfg[category])

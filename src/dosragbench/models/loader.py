"""Model loading with support for local GPU (4-bit) and Katana HPC (full precision)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dosragbench.utils.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """A loaded model with its tokenizer and config."""

    config: ModelConfig
    tokenizer: object
    model: object
    device: str

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
    ) -> dict:
        """Generate a response to a prompt. Returns dict with text + timing + token counts."""
        import time

        max_new_tokens = max_new_tokens or self.config.max_new_tokens

        # Format for chat models
        if self.config.chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Base models: use few-shot formatting
            input_text = prompt

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        input_token_count = inputs["input_ids"].shape[1]

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=None,
                top_p=None,
            )
        elapsed = time.perf_counter() - start

        # Decode only the newly generated tokens
        generated_tokens = outputs[0][input_token_count:]
        output_token_count = len(generated_tokens)
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return {
            "text": text,
            "latency_s": elapsed,
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "total_tokens": input_token_count + output_token_count,
        }

    def unload(self):
        """Release GPU memory."""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_model(config: ModelConfig, device: str = "auto") -> LoadedModel:
    """Load a model with appropriate quantization for the target device."""
    logger.info(f"Loading {config.name} ({config.hf_id})")

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set up quantization
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device if device != "cuda" else "auto",
        "trust_remote_code": True,
    }

    if config.quantization == "4bit" and device == "cuda":
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            # Don't set torch_dtype when using quantization_config
            model_kwargs.pop("torch_dtype", None)
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to bf16")

    elif config.quantization == "8bit" and device == "cuda":
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs.pop("torch_dtype", None)
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to bf16")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(config.hf_id, **model_kwargs)
    model.eval()

    logger.info(f"Loaded on {next(model.parameters()).device}")

    return LoadedModel(
        config=config,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


def detect_environment() -> str:
    """Detect whether we're running locally or on Katana HPC."""
    if os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID"):
        return "katana"
    return "local"

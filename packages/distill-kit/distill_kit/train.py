"""SFT and DPO with TRL, configured from a recipe YAML. Launched by scripts/train.sh.

A recipe is a full fine-tune under DeepSpeed ZeRO-3 (``deepspeed:``) where memory allows, or a LoRA adapter
(``lora:``) otherwise.

    accelerate launch --num_processes 8 --mixed_precision bf16 -m distill_kit.train recipes/sft-qwen3-8b.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

TRAINER_KEYS = {
    "learning_rate", "num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps",
    "warmup_ratio", "lr_scheduler_type", "weight_decay", "max_length", "bf16", "gradient_checkpointing",
    "logging_steps", "save_strategy", "seed", "beta", "max_prompt_length", "loss_type",
}
LORA_KEYS = {"r", "alpha", "dropout", "target_modules"}


def load_recipe(path: str | Path) -> dict:
    recipe = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("stage", "model", "data", "output_dir"):
        if key not in recipe:
            raise ValueError(f"recipe {path} is missing {key}")
    if recipe["stage"] not in {"sft", "dpo"}:
        raise ValueError("stage is sft or dpo")
    unknown = set(recipe.get("trainer", {})) - TRAINER_KEYS
    if unknown:
        raise ValueError(f"unknown trainer settings: {', '.join(sorted(unknown))}")
    if "lora" in recipe:
        unknown = set(recipe["lora"]) - LORA_KEYS
        if unknown:
            raise ValueError(f"unknown lora settings: {', '.join(sorted(unknown))}")
        if recipe.get("deepspeed"):
            raise ValueError("a recipe trains with LoRA or ZeRO-3 (full fine-tune), not both")
    return recipe


def peft_config(recipe: dict):
    """The LoRA adapter of a recipe, or None for a full fine-tune."""
    if "lora" not in recipe:
        return None
    from peft import LoraConfig

    lora = recipe["lora"]
    return LoraConfig(r=lora.get("r", 16), lora_alpha=lora.get("alpha", 32), lora_dropout=lora.get("dropout", 0.05),
                      target_modules=lora.get("target_modules", "all-linear"), task_type="CAUSAL_LM")


def to_chat(row: dict, template_prompt: str | None = None) -> dict:
    """An SFT example in chat format: the rendered prompt, then the teacher's JSON."""
    prompt = row.get("prompt") or template_prompt or row["document"]
    return {"messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": json.dumps(row["target"], ensure_ascii=False)}]}


def main(argv: list[str] | None = None) -> int:
    recipe = load_recipe((argv or sys.argv[1:])[0])
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(recipe["model"])
    model = AutoModelForCausalLM.from_pretrained(recipe["model"], torch_dtype="auto")
    data = load_dataset("json", data_files=recipe["data"], split="train")
    args = dict(recipe.get("trainer", {}), output_dir=recipe["output_dir"],
                deepspeed=recipe.get("deepspeed"), report_to=recipe.get("report_to", "none"))
    if recipe["stage"] == "sft":
        from trl import SFTConfig, SFTTrainer

        if "messages" not in data.column_names:
            from distill_kit.teacher import render

            schema = json.loads(Path(recipe["schema"]).read_text())
            data = data.map(lambda r: to_chat(r, render(r["document"], schema)))
        trainer = SFTTrainer(model=model, args=SFTConfig(**args), train_dataset=data, processing_class=tok,
                             peft_config=peft_config(recipe))
    else:
        from trl import DPOConfig, DPOTrainer

        trainer = DPOTrainer(model=model, args=DPOConfig(**args), train_dataset=data, processing_class=tok,
                             peft_config=peft_config(recipe))
    trainer.train()
    trainer.save_model(recipe["output_dir"])
    tok.save_pretrained(recipe["output_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

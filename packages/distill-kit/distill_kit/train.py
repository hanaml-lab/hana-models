"""SFT and DPO with TRL, configured from a recipe YAML. Launched by scripts/train.sh.

    accelerate launch --config_file recipes/accelerate.yaml -m distill_kit.train recipes/sft-qwen3-8b.yaml
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
    return recipe


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
        trainer = SFTTrainer(model=model, args=SFTConfig(**args), train_dataset=data, processing_class=tok)
    else:
        from trl import DPOConfig, DPOTrainer

        trainer = DPOTrainer(model=model, args=DPOConfig(**args), train_dataset=data, processing_class=tok)
    trainer.train()
    trainer.save_model(recipe["output_dir"])
    tok.save_pretrained(recipe["output_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

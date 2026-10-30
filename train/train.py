"""QLoRA SFT. python train/train.py train/config.yaml  (needs the `train` extra and a CUDA GPU)."""
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from text2sql.llm import messages

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())


def fmt(r):
    return {"messages": messages(r["question"]) + [{"role": "assistant", "content": r["sql"]}]}


ds = {s: load_dataset("json", data_files=f"data/{s}.jsonl")["train"] for s in ("train", "val")}
ds = {k: v.map(fmt, remove_columns=v.column_names) for k, v in ds.items()}

args = SFTConfig(
    output_dir=cfg["out"], seed=cfg["seed"], num_train_epochs=cfg["epochs"], learning_rate=cfg["lr"],
    per_device_train_batch_size=cfg["batch"], gradient_accumulation_steps=cfg["grad_accum"],
    max_length=cfg["max_len"], bf16=True, logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
    report_to="mlflow",
    model_init_kwargs={
        "torch_dtype": torch.bfloat16,
        "quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16),
    },
)
lora = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
                  target_modules="all-linear", task_type="CAUSAL_LM")
trainer = SFTTrainer(model=cfg["model"], args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
                     peft_config=lora)
t0 = time.time()
trainer.train()
trainer.save_model(cfg["out"])
# GPU hours for the README table
Path(cfg["out"], "train_stats.json").write_text(
    json.dumps({"gpu_hours": (time.time() - t0) / 3600, "gpu": torch.cuda.get_device_name(0)}))

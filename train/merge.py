"""Merge the LoRA adapter into the base model. python train/merge.py <base> <adapter_dir> <out_dir>"""
import sys

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

base, adapter, out = sys.argv[1:4]
AutoPeftModelForCausalLM.from_pretrained(adapter, torch_dtype=torch.bfloat16).merge_and_unload().save_pretrained(out)
AutoTokenizer.from_pretrained(base).save_pretrained(out)

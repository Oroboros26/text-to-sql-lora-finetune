NAME   ?= run
MODEL  ?= Qwen/Qwen2.5-Coder-1.5B-Instruct
SHOTS  ?= 0

.PHONY: run test eval data train merge table publish

run:    ## serve a model behind an OpenAI-compatible API (needs vllm): make run MODEL=...
	vllm serve $(MODEL) --port 8000

test:
	ruff check . && pytest

data:   ## needs LLM_BASE_URL, LLM_API_KEY, GEN_MODEL
	python -m text2sql.data

eval:   ## make eval NAME=qwen-base-0shot MODEL=... SHOTS=0
	python eval/run.py --name $(NAME) --model $(MODEL) --shots $(SHOTS)

table:
	python eval/table.py

train:
	python train/train.py train/config.yaml

merge:
	python train/merge.py $(MODEL) runs/qlora-v1 runs/merged

publish: ## after `huggingface-cli login`; set HF_USER
	huggingface-cli upload $(HF_USER)/text2sql-qwen-qlora runs/merged

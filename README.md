# Fine-tuning LLMs for Conversational Query Rewriting

Code for the IEEE Access 2025 article
**"ChatGPT Versus Modest Large Language Models: An Extensive Study on Benefits and Drawbacks for Conversational Search"**
(G. Rocchietti, C. Rulli, F. M. Nardini, C. I. Muntean, R. Perego, O. Frieder).

[Paper (IEEE Access, open access)](https://doi.org/10.1109/ACCESS.2025.3529741) ·
[Fine-tuning data on Hugging Face](https://huggingface.co/datasets/giuid/qrecc_raw_context)

## Overview

In conversational search, an utterance such as *"What about its side effects?"* cannot be
answered without the previous turns. Query rewriting turns it into a self-contained query
that a standard retrieval pipeline can process.

The article compares two ways of doing this with large language models:

1. prompting `gpt-3.5-turbo` (zero-, one- and few-shot);
2. fine-tuning smaller open models (FLAN-T5 large and XL, Falcon-7B, Mistral-7B, Llama-2 7B and 13B,
   and a 4-bit QLoRA version of Llama-2 70B) on the QReCC rewriting dataset.

Rewrites are evaluated by their effect on retrieval on TREC CAsT 2019 and 2020
(DPH first stage with PyTerrier, MonoT5 reranking of the top 1000).
The trade-off between effectiveness and cost is discussed in the article.

This repository contains the scripts used to build the fine-tuning data, fine-tune the
decoder-only models and generate rewrites for TREC CAsT.

## Repository structure

| File | Purpose |
|---|---|
| `manage_dataset.py` | Builds the QReCC fine-tuning files (context, question and rewrite separated by special tokens) and the TREC CAsT input files. |
| `methods.py` | Helpers to load TREC CAsT topics (2019–2021) into data frames. |
| `finetuning_script_new.py` | Full fine-tuning, or LoRA on a quantised model with `--quantize`. |
| `finetuning_script_qlora.py` | QLoRA fine-tuning (4- or 8-bit) for the largest models. |
| `run_inference_script_new.py` | Generates rewrites for TREC CAsT with a fine-tuned checkpoint. |
| `docker_1.dockerimage` | Dockerfile of the environment (based on `nvcr.io/nvidia/pytorch:22.12-py3`). |

## Input format

Each training example is a single string:

```
>>INTRODUCTION<< {previous questions} >>QUESTION<< {current question} >>ANSWER<< {rewrite}
```

The three special tokens come from the Falcon tokenizer and are used for all models.
At inference time the model receives the same string up to `>>ANSWER<<` and generates the rewrite.
The preprocessed dataset is available as
[`giuid/qrecc_raw_context`](https://huggingface.co/datasets/giuid/qrecc_raw_context);
TREC CAsT conversations were removed from QReCC to avoid overlap with the evaluation data.

## Usage

Requirements: Python 3.9+, PyTorch, `transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`, `pandas`.
Gated models (e.g. Llama-2) require a Hugging Face token: run `huggingface-cli login` or set
`HF_TOKEN` in the environment.

**Fine-tuning**

```bash
python finetuning_script_new.py \
    --model_name meta-llama/Llama-2-13b-hf \
    --dataset giuid/qrecc_raw_context \
    --context_length 128 --batch_size 32 --learning_rate 5e-6 --num_epochs 5 \
    --model_output_path models/llama-2-13b_qrecc

# 4-bit QLoRA for large models
python finetuning_script_qlora.py --model_name meta-llama/Llama-2-70b-hf --n_bits 4
```

**Rewriting TREC CAsT**

`--trec_path` must contain `trec19_for_rewriting.csv`, `trec20_for_rewriting.csv` and
`trec21_for_rewriting.csv` (columns `qid, conv_id, turn, raw_context, rewritten_context,
question, prompt_raw_context, prompt_rewritten_context`), produced by `manage_dataset.py`.

```bash
python run_inference_script_new.py \
    --model_name models/llama-2-13b_qrecc/checkpoint-XXXX \
    --trec_path data/trec/ \
    --year 0 \
    --output_path rewritings/llama-2-13b/ \
    --bnb --file_name quantized_4bit
```

`--year 0` rewrites CAsT 2019, 2020 and 2021; `--bnb` loads the model in 4-bit NF4.
For each year, the script writes a CSV with the raw and cleaned rewrites and a text file with
the average generation time per utterance.

Retrieval and evaluation of the rewrites follow the PyTerrier pipeline of
[hpclab/conv-llm](https://github.com/hpclab/conv-llm).

## Citation

```bibtex
@article{rocchietti2025chatgpt,
  author  = {Rocchietti, Guido and Rulli, Cosimo and Nardini, Franco Maria and Muntean, Cristina Ioana and Perego, Raffaele and Frieder, Ophir},
  title   = {{ChatGPT} Versus Modest Large Language Models: An Extensive Study on Benefits and Drawbacks for Conversational Search},
  journal = {IEEE Access},
  volume  = {13},
  pages   = {15253--15271},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.3529741}
}
```

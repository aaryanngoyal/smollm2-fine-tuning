# smollm2-sft: Fine-tuned SmolLM2-360M-Instruct on CodeAlpaca-20k

![GitHub stars](https://img.shields.io/github/stars/your-username/smollm2-sft?style=social)
![GitHub forks](https://img.shields.io/github/fortunes/your-username/smollm2-sft?style=social)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A LoRA (Low-Rank Adaptation) fine-tuning pipeline for `SmolLM2-360M-Instruct` on the `CodeAlpaca-20k` instruction dataset.

## Quick Overview

This repository contains:
- **Training script** (`smollm2-fine-tuning.py`) — Full LoRA fine-tuning pipeline
- **Fine-tuned model** — LoRA adapters saved in `smollm2-sft/`
- **Model card** — Rendered on Hugging Face Hub: `your-username/smollm2-sft`
- **Reproducibility** — `requirements.txt` and data preprocessing documented

## Repository Structure

```
/smollm2-sft/ (Git root)
├── smollm2-fine-tuning.py    ← Training pipeline
├── README.md                 ← This file
├── requirements.txt          ← Python dependencies
├── .gitignore                ← Git ignores
└── smollm2-sft/              ← PEFT output directory (LoRA adapters) (gitignored)
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── chat_template.jinja
    ├── trainer_state.json
    └── training_args.bin
```

## Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/smollm2-sft.git
cd smollm2-sft
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```txt
torch>=2.3.0
transformers>=4.36.0
peft>=0.9.0
datasets>=2.14.0
```

### 3. (Optional) Login to Hugging Face Hub

If you want to push the model yourself:

```bash
huggingface-cli login
```

## Running the Training Pipeline

Execute the fine-tuning script:

```bash
python smollm2-fine-tuning.py
```

**Key training hyperparameters** (configured in `smollm2-fine-tuning.py`):

| Parameter | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| Dataset | `sahil2801/CodeAlpaca-20k` (train[:1000] used by default) |
| LoRA rank (`r`) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Learning rate | 1e-4 |
| Batch size | 2 per device |
| Gradient accumulation | 4 |
| Max steps | 250 |
| Mixed precision | `bf16` |

> **Note:** The script uses only `train[:1000]` for quick iteration. To train on the full 20k set, change `split="train[:1000]"` to `split="train"` in `smollm2-fine-tuning.py`.

## Using the Fine-Tuned Model

After training, the PEFT adapters are in `smollm2-sft/`. To load and generate with the model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype="auto",
    device_map="cpu",
)

# Load LoRA adapters
model = PeftModel.from_pretrained(model, "smollm2-sft")

# Generate text
prompt = "Write a Python function to reverse a string."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Hugging Face Hub

The fine-tuned model is also published on the Hugging Face Hub, where it can be loaded directly:

```
https://huggingface.co/your-username/smollm2-sft
```

Load it with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("your-username/smollm2-sft")
# Or load PEFT separately...
```

## License

- **Base model** (`SmolLM2-360M-Instruct`): Apache-2.0
- **Dataset** (`CodeAlpaca-20k`): MIT
- **This repository**: MIT

## Citation

If you use this model or code, please cite the base model and dataset:

```bibtex
@article{smolLM2,
  author = {HuggingFaceTB},
  title = {SmolLM2: Pure On-device Language Models},
  year = {2024},
  publisher = {Hugging Face}
}

@dataset{CodeAlpaca-20k,
  author = {Sahil2801},
  title = {CodeAlpaca-20k},
  year = {2023},
  publisher = {Hugging Face Datasets}
}
```

## Contact

Created by **aaryangoyal**. 

---
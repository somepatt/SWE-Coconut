# COCONUT Code LLM

Training code LLMs with **Chain of Continuous Thought** (COCONUT) for bug fixing tasks on SWE-bench.

## Features

✨ **COCONUT Multi-stage Training**
- Stage 0: Language-based Chain of Thought
- Stage N: Gradual replacement with latent reasoning
- Improved reasoning without token overhead

✨ **Production-Ready Code**
- Clean architecture with separation of concerns
- Comprehensive logging throughout
- Easy hyperparameter management via YAML config
- Type hints and docstrings

✨ **Optimizations**
- 4-bit quantization with bitsandbytes
- LoRA for efficient fine-tuning
- FlashAttention-2 support
- Gradient accumulation

## Installation

pip install -r requirements.txt

## Configuration

Edit `config/default.yaml`:

model:
name: "Qwen/Qwen3-0.6B"
use_quantization: true
use_lora: true

training:
num_stages: 3
latent_dim: 256
continuous_thought_steps: 4

optimizer:
lr: 5e-5
weight_decay: 0.01

## Training

python scripts/train.py

## Evaluation

python scripts/evaluate.py

## Project Structure

- `config/` — Configuration files
- `src/` — Main source code
  - `config.py` — Config management
  - `model.py` — COCONUT model
  - `data.py` — Dataset loading & preprocessing
  - `trainer.py` — Training loop
  - `optimizer.py` — Optimizer & scheduler
  - `logger.py` — Logging setup
  - `utils.py` — Utilities
- `scripts/` — Training & evaluation
- `tests/` — Unit tests

## Logging

All events logged to:
- Console (INFO level)
- `outputs/{experiment_name}.log` (DEBUG level)
- W&B (if enabled)

## Paper Reference

- COCONUT: Training Large Language Models to Reason in a Continuous Latent Space
- Meta AI Research, 2024

## License

MIT

## Как использовать:
bash
# 1. Клонируйте структуру
mkdir -p coconut-code-llm/{src,config,scripts,tests,outputs}

# 2. Скопируйте файлы
cp config/default.yaml coconut-code-llm/config/
cp src/*.py coconut-code-llm/src/
cp scripts/*.py coconut-code-llm/scripts/
cp requirements.txt coconut-code-llm/

# 3. Установите зависимости
cd coconut-code-llm
pip install -r requirements.txt

# 4. Запустите тренировку
python scripts/train.py
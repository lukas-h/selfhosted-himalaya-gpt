---
license: other
license_link: https://huggingface.co/himalaya-ai/himalayagpt-0.5b-it
language:
- en
- ne
- hi
pipeline_tag: text-generation
base_model: himalaya-ai/himalayagpt-0.5b-it
base_model_relation: quantized
tags:
- gguf
- llama-cpp
- nanochat
- requires-fork
- quantized
quantized_by: lukashimsel
inference: false
---

# himalayagpt-0.5b-it — GGUF

GGUF quantizations of [`himalaya-ai/himalayagpt-0.5b-it`](https://huggingface.co/himalaya-ai/himalayagpt-0.5b-it) by the [Himalaya AI](https://himalayaai.org/) team.

For everything about the model itself — architecture, training, intended use, limitations, languages — see the **[base model card](https://huggingface.co/himalaya-ai/himalayagpt-0.5b-it)**. This page only documents the quantization.

## ⚠️ Custom runtime required

These GGUFs **do not run on upstream `ggml-org/llama.cpp`**. Nanochat is not a supported architecture upstream — vanilla llama.cpp will reject the file with `Model NanochatForCausalLM is not supported`.

Use the fork: **[`HimalayaAI/llama.cpp`](https://github.com/HimalayaAI/llama.cpp)**.

LM Studio, Ollama, Open WebUI, and other UIs that bundle vanilla llama.cpp will not work until upstream ships nanochat support.

## Files

| File | Size | Notes |
|---|---|---|
| `himalayagpt-0.5b-it-bf16.gguf` | 1.0 GB | Recommended. Numerically equivalent to the HF Transformers reference. |
| `himalayagpt-0.5b-it-Q8_0.gguf` | 533 MB | Drop-in for BF16; near-identical quality. |
| `himalayagpt-0.5b-it-Q4_K_M.gguf` | 300 MB | Smallest. Math accuracy notably worse than BF16/Q8_0. |

> **Don't build an F16 GGUF.** The model's squared-ReLU MLP overflows F16's ±65504 range and yields NaN logits. BF16 has the same on-disk size with F32 dynamic range — use that instead.

The chat template (`<|user_start|>…<|user_end|><|assistant_start|>…<|assistant_end|>`) and EOT token are embedded in the GGUF metadata.

OpenAI-style `system` messages are supported and merge into the next user turn (separated by a blank line). Tool/`developer` roles render as user content. The embedded template also strips the 9 nanochat control-token literal strings from message content to prevent prompt injection — if you want a literal `<|assistant_start|>` in user input, that will not survive into the rendered prompt by design.

## How to use

Build the fork, then run `llama-cli`:

```bash
git clone --recursive https://github.com/lukas-h/llama.cpp.git
cd llama.cpp
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build build --target llama-cli -j$(nproc)

./build/bin/llama-cli \
  -m himalayagpt-0.5b-it-Q8_0.gguf \
  --jinja --temp 0.2 --top-k 40 --repeat-penalty 1.08 -n 256
```

Recommended sampling (matches the upstream Himalaya AI Colab): `temperature=0.2, top_k=40, repetition_penalty=1.08`. Greedy (`temp=0`) loops on long answers.

Full build/deploy notes: <https://github.com/lukas-h/selfhosted-himalaya-gpt>.

## Numerical fidelity vs base model

Verified against the upstream Hugging Face Transformers implementation:

- Tokenizer: byte-identical token IDs.
- Prefill logits (F32 GGUF, 9-token prompt): cosine similarity = 1.000000, max abs diff < 0.025 across all positions.
- Greedy autoregressive decode (16 steps): byte-identical token sequence to the HF reference, per-step cosine ≥ 0.9999.

## Credits

- **Model**: the [Himalaya AI](https://himalayaai.org/) team — see the [base model card](https://huggingface.co/himalaya-ai/himalayagpt-0.5b-it).
- **Quantization & runtime patches**: — see <https://github.com/HimalayaAI/llama.cpp> and <https://github.com/lukas-h/selfhosted-himalaya-gpt>.

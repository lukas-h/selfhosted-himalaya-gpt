# Nanochat GGUF / llama.cpp Handover

Date: 2026-05-07

## Status

**Done.** `himalaya-ai/himalayagpt-0.5b-it` runs in the local llama.cpp fork at numerical parity with HF Transformers (cosine similarity = 1.0, max abs logit diff < 0.025 across 9 prefill positions). Working chat-formatted generation is verified. BF16, Q8_0 and Q4_K_M GGUFs all produce sensible output through the full chat path.

## Artifacts

In `/home/lukashimsel/Projects/scalabs/himalaya/`:

| File | Size | SHA-256 |
|------|------|---------|
| `himalayagpt-0.5b-it-bf16.gguf` | 1.0 GB | `dbdebb8119e060ef2276ff975da2058eb97ad719d9e367864821ae85411a39cc` |
| `himalayagpt-0.5b-it-Q8_0.gguf` | 533 MB | `9e72cb8ae0316510211b5d8fb638afc5561397d8d08d3dd0ce57bf62795d1c98` |
| `himalayagpt-0.5b-it-Q4_K_M.gguf` | 300 MB | `409984b53fded0e3fefc5d06647b908755076c43e8221f296b38f812d2ad8bf8` |

The previously-shipped F16 GGUF was deleted because **F16 is broken for this model** — the squared-ReLU MLP produces activations > 65,504 and overflows F16 dynamic range, yielding NaN logits. Use BF16 (same size as F16, F32 exponent range) or any K/Q quant.

## Code in the fork

`/home/lukashimsel/Projects/scalabs/himalaya/llama.cpp` (commit `97f06e9`, uncommitted patches in working tree):

- `convert_hf_to_gguf.py` — `NanochatModel(TextModel)` for `NanochatForCausalLM`. Adds GGUF KV (rope base, rms eps, BOS/EOS, `add_bos_token=True`), exports the pickled tiktoken (`rustbpe`) tokenizer as GPT-2-style BPE with `tokenizer.ggml.pre = "nanochat"`, embeds a Jinja chat template using `<|user_start|>…<|user_end|><|assistant_start|>…<|assistant_end|>`.
- `gguf-py/gguf/constants.py` — `MODEL_ARCH.NANOCHAT` plus the seven nanochat-specific tensor constants (`RESID_LAMBDAS`, `X0_LAMBDAS`, `SMEAR_GATE`, `SMEAR_LAMBDA`, `BACKOUT_LAMBDA`, `VALUE_EMBD`, `VE_GATE`).
- `src/llama-arch.{cpp,h}` — `LLM_ARCH_NANOCHAT` plus per-tensor `LLM_TENSOR_NANOCHAT_*` registrations.
- `src/llama-vocab.{cpp,h}` — `LLAMA_VOCAB_PRE_TYPE_NANOCHAT = 51` and the nanochat regex (possessive quantifiers rewritten as greedy).
- `src/llama-model.{cpp,h}` — factory dispatch and per-arch / per-layer tensor pointers; `LLM_ARCH_NANOCHAT` uses `LLAMA_ROPE_TYPE_NEOX`.
- `src/models/models.h` — declares `llama_model_nanochat`.
- `src/models/nanochat.cpp` — runtime backend.

Patch summary: `9 files changed, 217 insertions(+), 1 deletion(-)`.

## Critical findings

1. **Nanochat trains with `R(-pos·θ)` rotation, not standard `R(+pos·θ)`.** Karpathy's `_apply_rotary` is `y1 = x1·cos + x2·sin; y2 = -x1·sin + x2·cos`, which is the inverse rotation of GGML's NEOX rope. We compensate at the call site in `nanochat.cpp` by passing `freq_scale = -freq_scale` to `ggml_rope_ext`, which negates the precomputed sin while leaving cos alone (cos is even, sin is odd). Confirmed equivalent to HF; without it, pos-1 logits diverge with cos≈0.9988 / max-abs ≈ 7.4 vs the corrected cos = 1.000000 / max-abs ≈ 0.012.

2. **F16 overflows because the model uses squared-ReLU MLPs.** `c_proj(relu(c_fc(x)).square())` produces intermediate magnitudes ~10⁵ which exceeds F16's ±65504. We never store activations as F16, but ggml's rms_norm path appears to materialize squared sums in F16 somewhere downstream and overflows to NaN. BF16 has the same 16-bit storage with F32-equivalent exponent range and works correctly. K/Q quants also work because their dequant path lands in F32.

3. **HF `from_pretrained` clobbers the rotary buffers.** `NanochatBackbone.cos` and `.sin` are registered with `persistent=False`, computed during `__init__`, then zeroed by accelerate-style empty-init that runs after weight loading. Any HF reference comparison must call `model.model._refresh_rotary()` after `from_pretrained` or you'll be comparing against a broken HF forward (which is what produced the "model output is poor/repetitive" claim in the previous handover — that was HF's own bug, not the port's).

4. **`Vcur_value_embd-0` cb name doesn't show up in the eval callback** even though `cb()` is called. Either ggml fuses the `ggml_add` or the cb call is overwritten downstream. Doesn't affect correctness — final logit parity holds — but means you can't filter on that name when debugging.

## Verification done

- **Tokenizer parity**: HF `enc.encode_ordinary` and llama-tokenize give identical token IDs across multiple prompts (Devanagari, English, ASCII).
- **Prefill logits parity**: 9-token prompt, F32 GGUF, all 9 positions: `cos=1.000000, max_abs<0.025, top1_match=True`. Verified intermediate parity at `inp_smear`, `layer_resid_mix-0`, `attn_norm-0`, `Qcur_normed-0`, `Kcur_normed-0`. Caveat: we only verified prefill at the same context; autoregressive single-token decode is structurally different (smear branch is skipped because `n_tokens == 1`) and still hasn't been compared against HF's full-recompute path.
- **End-to-end chat generation**: BF16, Q8_0, Q4_K_M all produce coherent chat replies. EOT (`<|assistant_end|>`) registered, so generation stops cleanly at turn end.
- **Autoregressive parity**: 11-token chat prompt, greedy decode for 16 steps. HF (full recompute every step, smear active) and llama.cpp (KV-cache, smear skipped at `n_tokens==1`) produce **byte-identical token sequences** (`[302, 1565, 469, 278, 282, 389, 8671, 401, 282, 278, 12305, 585, 260, 317, 564, 32763]`). Per-step cosine similarity stays ≥0.9999, max abs logit diff ≤0.29 across all 16 generation steps. The smear-skip drift is real but too small to change argmax in practice — this closes the open question from the previous handover.
- **Colab parity**: Ran the 5 prompts from the upstream Colab (`himalaya_gpt_test.ipynb`) through BF16 with their recommended sampling (`temp=0.2, top_k=40, rep_pen=1.08`). Prompt 1 (Nepal capital → "नेपालको राजधानी काठमाडौं हो।") is byte-identical to Colab output. Other prompts produce different wording due to RNG-implementation differences between PyTorch and llama.cpp samplers, but topic, language, and structure all match. Math accuracy is shaky on both implementations (Colab got 17×19=323 right, ours sometimes gives 303/340 with different seeds; the model just isn't a math model).
- **Repository revision**: Tested both `22f823e0…` and the current `main` (`17ec36f8…`); weights are byte-identical across the two, only `README.md`/`RELEASE_HOTFIX.json`/`run_standalone_inference.py` differ. GGUFs in this repo were converted from `17ec36f8…`.

## Build commands

```bash
cd /home/lukashimsel/Projects/scalabs/himalaya/llama.cpp

# venv has cmake + ninja installed (uv pip install)
PATH="/home/lukashimsel/Projects/scalabs/himalaya/.venv/bin:$PATH" \
  ../.venv/bin/cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DGGML_NATIVE=ON

PATH="/home/lukashimsel/Projects/scalabs/himalaya/.venv/bin:$PATH" \
  ../.venv/bin/cmake --build build \
    --target llama-cli llama-tokenize llama-completion llama-quantize \
    -j$(nproc)
```

Note: this fork gates `tools/cli/` behind `LLAMA_BUILD_SERVER=ON`. Without that, `llama-cli` is silently skipped from the build.

## Conversion + quantization

```bash
# BF16 (recommended source for quants)
../.venv/bin/python convert_hf_to_gguf.py \
  /home/lukashimsel/.cache/huggingface/hub/models--himalaya-ai--himalayagpt-0.5b-it/snapshots/22f823e07df6e6067f26108c3abfa83489dfe63e \
  --outfile ../himalayagpt-0.5b-it-bf16.gguf --outtype bf16

# Q8_0 / Q4_K_M — must override the small non-block-aligned tensors,
# else quantize fails on nanochat_ve_gate (12 cols, not div by 32).
./build/bin/llama-quantize \
  --tensor-type nanochat_ve_gate=f16 \
  --tensor-type nanochat_smear_gate=f16 \
  --tensor-type nanochat_smear_lambda=f32 \
  --tensor-type nanochat_backout_lambda=f32 \
  --tensor-type nanochat_resid_lambdas=f32 \
  --tensor-type nanochat_x0_lambdas=f32 \
  ../himalayagpt-0.5b-it-bf16.gguf ../himalayagpt-0.5b-it-Q4_K_M.gguf Q4_K_M
```

## Running

This fork's `llama-cli` is interactive-only when a chat template is present. For one-shot completion use `llama-completion` with `-no-cnv`. Pass `--special` so prompt-side `<|user_start|>…<|assistant_start|>` get tokenized as actual special-token IDs (32760, 32761, 32762) — without it they fall back to byte-level tokenization which the model partially recovers from but quality drops. **Don't add an explicit `<|bos|>`** to the prompt — `add_bos_token=True` is in the GGUF, so llama.cpp prepends it automatically; doubling it is harmless but wastes context.

```bash
./build/bin/llama-completion \
  -m ../himalayagpt-0.5b-it-bf16.gguf \
  -p '<|user_start|>What is the capital of Nepal?<|user_end|><|assistant_start|>' \
  --special -no-cnv -n 256 \
  --temp 0.2 --top-k 40 --repeat-penalty 1.08 --seed 42
```

`<|assistant_end|>` (32763) is registered as the EOT token, so generation stops cleanly at end-of-turn.

**Don't pass `--jinja` when your prompt is already chat-formatted** — it wraps the literal prompt as a user turn and you end up with nested `<|user_start|>` markers. `--jinja` is for the case where you pass plain text and want llama.cpp to render it through the embedded chat template. LM Studio and similar UIs handle this correctly through their own chat plumbing.

Sampling settings the upstream Colab uses and that match the model's training distribution: `temperature=0.2, top_k=40, repetition_penalty=1.08`. Greedy (`temp=0`) is fine for short factual questions but loops on long responses (the squared-ReLU MLP makes the model surprisingly attractor-prone).

## Open work

- **Optional cleanup.** The `Vcur_value_embd` cb name not surfacing in eval callback is a minor debugging nuisance, not a correctness issue.
- **Upstream PR.** Per llama.cpp's `AGENTS.md`, the upstream project does not accept AI-generated changes; "Private forks are exempt" so the user's `lukas-h/llama.cpp` fork is the right home for these patches. Don't attempt to upstream as-is.

## Things to keep in mind for the next session

- The handover doc's "model output was poor/repetitive on CPU Transformers" line was misleading — that was HF's own broken rotary buffer, not the model. After `_refresh_rotary()` HF generates the same output as our port.
- The lambda mixing has surprisingly large coefficients (`x0_lambdas[0] = 254.7`). Tiny upstream errors get amplified ~250x at layer 0. This is by design (Karpathy speedrun-style architecture) but means F16 precision is genuinely insufficient — don't be tempted to ship F16.
- `add_bos_token = True` is set in the GGUF, so chat templates that don't render `<|bos|>` themselves will still get one prepended by the tokenizer. If you render with `<|bos|>` literally (as I did in some debug commands) you get a double BOS — harmless for this model but worth knowing.

## Useful debug commands

Logits dumper (custom): `/tmp/dump_logits.cpp` — reads a comma-separated token list, dumps prefill logits and a configurable filter of intermediate tensors for callback-based comparison against HF.

```bash
# Build (requires libllama.so from the fork's build/)
g++ -std=c++17 -O2 \
  -I/home/lukashimsel/Projects/scalabs/himalaya/llama.cpp/include \
  -I/home/lukashimsel/Projects/scalabs/himalaya/llama.cpp/ggml/include \
  /tmp/dump_logits.cpp \
  -L/home/lukashimsel/Projects/scalabs/himalaya/llama.cpp/build/bin \
  -lllama -lggml -lggml-base -lggml-cpu -lpthread -lm \
  -Wl,-rpath,/home/lukashimsel/Projects/scalabs/himalaya/llama.cpp/build/bin \
  -o /tmp/dump_logits

# HF reference (must include _refresh_rotary)
../.venv/bin/python -c "
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('<snapshot>', trust_remote_code=True).eval()
m.model._refresh_rotary()  # critical
ids = [302, 1565, 3471, 295, 389, 8671, 401, 282, 32]
with torch.no_grad():
    print(m(torch.tensor([ids])).logits[0, -1].argmax().item())
"
```

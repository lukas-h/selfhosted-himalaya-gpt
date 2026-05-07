# selfhosted-himalaya-api

Run [`himalaya-ai/himalayagpt-0.5b-it`](https://huggingface.co/himalaya-ai/himalayagpt-0.5b-it) — the open-weight model from the [**Himalaya AI**](https://himalayaai.org/) team (Karpathy nanochat-architecture, ~0.5B params, instruction-tuned, English + Nepali/Hindi) — locally as a GGUF model in any llama.cpp-compatible runtime: `llama-cli`, LM Studio, Ollama, etc.

The upstream llama.cpp does **not** support nanochat models. This repo bundles a custom fork ([`lukas-h/llama.cpp`](https://github.com/lukas-h/llama.cpp)) with nanochat support added (custom RoPE direction, input smear, alternating value embeddings, squared-ReLU MLP, mid-layer backout, output softcap), plus prebuilt GGUFs in three precisions.

## What's in this repo

```
himalayagpt-0.5b-it-bf16.gguf      1.0 GB   recommended
himalayagpt-0.5b-it-Q8_0.gguf      533 MB   smaller, near-identical quality
himalayagpt-0.5b-it-Q4_K_M.gguf    300 MB   smallest, math accuracy notably worse
llama.cpp/                         submodule → lukas-h/llama.cpp
api/  (api branch only)            self-hosted OpenAI-compatible API (master + worker)
NANOCHAT_GGUF_HANDOVER.md          design notes, parity verification, internals
```

GGUF files are tracked via Git LFS.

The `api/` directory lives on the **`api` branch** (`git checkout api`). It contains the master FastAPI service, the GPU-side worker, and a small SSE mock for plumbing tests. See [`api/master/README.md`](api/master/README.md) and [`api/worker/README.md`](api/worker/README.md).

---

## 1. Clone the repo

```bash
git clone --recurse-submodules git@github.com:lukas-h/selfhosted-himalaya-api.git
cd selfhosted-himalaya-api
```

If you forgot `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

If you don't have Git LFS yet:

```bash
sudo apt install -y git-lfs   # Debian/Ubuntu
git lfs install
git lfs pull                  # downloads the .gguf files
```

---

## 2. Compile the llama.cpp fork

### Install build tools

```bash
sudo apt install -y build-essential cmake ninja-build
```

### Configure and build

```bash
cd llama.cpp

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DGGML_NATIVE=ON

cmake --build build --target llama-cli llama-completion llama-tokenize llama-quantize -j$(nproc)
```

This produces binaries under `llama.cpp/build/bin/`.

> Note: this fork gates `tools/cli/` behind `LLAMA_BUILD_SERVER=ON`. Without that flag, `llama-cli` is silently skipped from the build.

### Run interactive chat

```bash
cd llama.cpp     # (or use absolute paths for -m)

./build/bin/llama-cli \
  -m ../himalayagpt-0.5b-it-bf16.gguf \
  --jinja \
  --temp 0.2 --top-k 40 --repeat-penalty 1.08 \
  -n 256
```

Type at the `>` prompt. In-session commands: `/exit`, `/regen`, `/clear`, `/read <file>`, `/glob <pattern>`.

To pick a smaller quant just swap the `-m` argument to `Q8_0.gguf` or `Q4_K_M.gguf`.

### Run one-shot completion (no interactive REPL)

```bash
./build/bin/llama-completion \
  -m ../himalayagpt-0.5b-it-bf16.gguf \
  -p '<|user_start|>What is the capital of Nepal?<|user_end|><|assistant_start|>' \
  --special -no-cnv -n 256 \
  --temp 0.2 --top-k 40 --repeat-penalty 1.08
```

Generation stops automatically at `<|assistant_end|>`.

### Use with LM Studio / Ollama / other GGUF tools

Point them at any of the `.gguf` files. The chat template and end-of-turn token are embedded in the GGUF metadata so the host UI's chat panel will work without extra configuration.

---

## 3. (Re)convert the model from Hugging Face yourself

Skip this section if you just want to run the prebuilt GGUFs above. This is for re-converting after upstream changes or experimenting with quantization settings.

### Install the Python deps

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -U \
  "transformers==4.57.2" \
  "accelerate>=1.6.0" \
  "huggingface_hub>=0.23.0" \
  "safetensors>=0.4.5" \
  "torch" \
  "numpy" \
  "sentencepiece" \
  "protobuf"
```

### Download the model from Hugging Face

```bash
huggingface-cli download himalaya-ai/himalayagpt-0.5b-it
```

The files land under `~/.cache/huggingface/hub/models--himalaya-ai--himalayagpt-0.5b-it/snapshots/<sha>/`. Note that `<sha>` — the next step needs the path.

```bash
SNAP=$(ls -d ~/.cache/huggingface/hub/models--himalaya-ai--himalayagpt-0.5b-it/snapshots/*/ | head -1)
echo "$SNAP"
```

### Convert to GGUF (BF16 recommended)

> **Don't use F16** for this model. The squared-ReLU MLP produces activations that overflow F16's ±65504 range and yields NaN logits. BF16 has the same 16-bit storage size with F32-equivalent dynamic range and works correctly.

From inside the cloned repo:

```bash
cd llama.cpp

python convert_hf_to_gguf.py "$SNAP" \
  --outfile ../himalayagpt-0.5b-it-bf16.gguf \
  --outtype bf16
```

### Quantize from BF16 (optional)

The nanochat-specific small tensors (`nanochat_smear_gate`, `nanochat_ve_gate`, etc.) can't be block-quantized because their column counts aren't multiples of 32. Override their types so quantization succeeds:

```bash
./build/bin/llama-quantize \
  --tensor-type nanochat_ve_gate=f16 \
  --tensor-type nanochat_smear_gate=f16 \
  --tensor-type nanochat_smear_lambda=f32 \
  --tensor-type nanochat_backout_lambda=f32 \
  --tensor-type nanochat_resid_lambdas=f32 \
  --tensor-type nanochat_x0_lambdas=f32 \
  ../himalayagpt-0.5b-it-bf16.gguf \
  ../himalayagpt-0.5b-it-Q8_0.gguf \
  Q8_0
```

Replace `Q8_0` with `Q4_K_M` for the smaller variant. Same overrides apply.

---

## Notes on output quality

It's a 0.5B model — coherent but makes factual mistakes. Output matches the upstream Hugging Face reference numerically (see `NANOCHAT_GGUF_HANDOVER.md`). Recommended sampling: `temperature=0.2, top_k=40, repetition_penalty=1.08`.

---

## Self-hosting it as an OpenAI-compatible API

The `api` branch (`git checkout api`) ships a small two-piece stack you can deploy yourself, so client SDKs (`openai`, `litellm`, etc.) can talk to your home GPU over plain HTTPS without exposing the GPU to the internet.

```
       OpenAI SDK
            │ HTTPS
            ▼
   ┌────────────────────┐
   │ master (FastAPI)   │   tiny ($5 VPS, anywhere with a public IP)
   │ /v1/chat/completions │ token auth · rate limit · in-memory queue
   └────────────────────┘
            ▲
            │ workers long-poll  (outbound only)
   ┌────────────────────┐
   │ worker (GPU host)  │   your home machine / lab / wherever
   │ llama-server SYCL  │   pulls jobs, runs the model, streams back
   │ + agent (puller)   │   no inbound port needed
   └────────────────────┘
```

Minimum hardware:
- **1× $5 VPS** with a public IP and a domain pointing at it (Hetzner CX22 / DigitalOcean $4 / Vultr / etc.). Runs the master FastAPI; needs no GPU.
- **1× worker host** with an Intel Arc / NVIDIA / AMD GPU (≥4 GB VRAM is plenty). Internet access for outbound HTTPS only — Cloudflare Tunnel, NAT, or a normal home connection all work.

Each component is one `docker compose up` and a handful of env vars. **Full guide:** [`api/HOSTING.md`](api/HOSTING.md).

```bash
# master (on the VPS)
git clone -b api https://github.com/lukas-h/selfhosted-himalaya-gpt.git
cd selfhosted-himalaya-gpt/api/master
cp .env.example .env   # set API_TOKENS and WORKER_TOKEN
docker compose up -d

# worker (on the GPU host)
cd selfhosted-himalaya-gpt/api/worker
cp .env.example .env   # set MASTER_BASE_URL, WORKER_TOKEN, LLAMA_INTERNAL_KEY
docker compose up -d   # GGUFs auto-download on first boot
```

The worker has a `models-init` step that pulls the GGUFs from Hugging Face on first boot (override `MODELS_DOWNLOAD_BASE_URL`), or you can point `MODELS_HOST_PATH` at a directory you've populated yourself.

---

## Credits

- **Model**: [Himalaya AI](https://himalayaai.org/) — they trained `himalayagpt-0.5b-it` and released the weights and tokenizer under the [`himalaya-ai/himalayagpt-0.5b-it`](https://huggingface.co/himalaya-ai/himalayagpt-0.5b-it) Hugging Face repo. All credit for the model itself goes to them.
- **Architecture**: based on Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat) speedrun-GPT design.
- **Inference runtime**: built on [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp); the nanochat-specific runtime/converter patches in this repo's [`lukas-h/llama.cpp`](https://github.com/lukas-h/llama.cpp) fork are the only addition.

This repo only does the GGUF packaging and llama.cpp porting work — without the Himalaya AI team's open release, none of it would exist.

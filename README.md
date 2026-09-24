# Semantic Caching Layer for Local LLMs

A free, runs-on-your-laptop caching proxy that sits between your app and a
local LLM. It recognizes when a new question means the same thing as one
you already asked — even if it's worded differently — and answers instantly
from cache instead of re-running the model.

> **100% free to run.** No OpenAI key, no paid vector database, no cloud
> bill. Every tool below is free and open-source, and everything runs on
> your own machine.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [How It Works (Plain-English Explanation)](#2-how-it-works-plain-english-explanation)
3. [Architecture Overview (Technical)](#3-architecture-overview-technical)
4. [Tech Stack — and Why Every Piece Is Free](#4-tech-stack--and-why-every-piece-is-free)
5. [Installation](#5-installation)
6. [Usage](#6-usage)
7. [Project Structure](#7-project-structure)
8. [Configuration Reference](#8-configuration-reference)
9. [Testing](#9-testing)
10. [Optional: Full Monitoring Stack (Docker)](#10-optional-full-monitoring-stack-docker)
11. [Pushing This Project to GitHub](#11-pushing-this-project-to-github)
12. [Contributing](#12-contributing)
13. [Troubleshooting](#13-troubleshooting)
14. [License](#14-license)

---

## 1. Project Overview

**What it is:** A drop-in "middleware" that sits in front of a local LLM
(run through [Ollama](https://ollama.com), which is free). Any app that
sends chat requests can send them to this proxy instead, and the proxy
decides whether to answer instantly from a saved cache or forward the
question to the LLM.

**Who this is for:**
- **Beginner/intermediate developers** who want a real, working project to
  learn from — one that touches APIs, embeddings, vector search, caching
  strategy, and monitoring, without needing to pay for anything.
- **Non-technical readers** who want to understand *why* this kind of
  system matters (see the plain-English section below) and what problem
  it solves.

**Why it matters:** Every team running LLMs at scale eventually notices
the same thing: many incoming questions are near-duplicates of questions
already answered ("What is Python?" vs. "Explain Python to me"). Re-running
the full model for each one wastes time and money. A semantic cache — one
that compares *meaning*, not exact text — fixes that.

---

## 2. How It Works (Plain-English Explanation)

Imagine a librarian who has answered thousands of reference questions.
When someone asks *"What is Python used for?"*, and later someone else
asks *"Tell me what people use Python for"* — a good librarian recognizes
these are the same question and gives the same answer immediately,
instead of researching it again from scratch.

This project is that librarian, built in software:

1. **It reads the meaning of a question**, not just its exact wording,
   using a small AI model that turns text into a list of numbers (an
   "embedding") capturing what the sentence *means*.
2. **It compares that meaning to everything it has answered before.** If
   something very similar (above a configurable similarity score) was
   already answered *recently enough* (answers can "expire" after a
   while, just like a librarian wouldn't reuse yesterday's weather
   report), it reuses that answer instantly.
3. **If nothing matches, it asks the real LLM**, saves the new
   question-and-answer pair for next time, and returns the answer.

The result: repeated or rephrased questions get answered in milliseconds
instead of seconds, and the LLM is only invoked for genuinely new
questions.

---

## 3. Architecture Overview (Technical)

```
                     ┌─────────────────────────────┐
                     │   Your App / curl / script   │
                     └───────────────┬──────────────┘
                                     │ POST /v1/chat/completions
                                     ▼
                     ┌─────────────────────────────┐
                     │   FastAPI Caching Proxy      │
                     │   (src/main.py)              │
                     └───────────────┬──────────────┘
                                     │
                 ┌───────────────────┼───────────────────┐
                 ▼                                       ▼
   ┌─────────────────────────┐            ┌───────────────────────────┐
   │ 1. Embed the prompt      │            │ 2. Query the vector cache │
   │ (sentence-transformers,  │──────────▶ │ (ChromaDB, local files)   │
   │  free, local, offline)   │            │ namespace-isolated,       │
   └─────────────────────────┘            │ TTL-aware similarity      │
                                            │ search                    │
                                            └─────────────┬─────────────┘
                                                           │
                                     ┌─────────────────────┴────────────────────┐
                                     ▼ Cache HIT                                ▼ Cache MISS
                        ┌─────────────────────────┐             ┌─────────────────────────────┐
                        │ Return saved answer      │             │ Forward to Ollama (free,    │
                        │ instantly (no LLM call)  │             │ local LLM server), then     │
                        └─────────────────────────┘             │ store the new answer for    │
                                                                  │ next time                    │
                                                                  └─────────────────────────────┘
                                     │
                                     ▼
                     ┌─────────────────────────────┐
                     │ Prometheus metrics (/metrics)│ ── optional ──▶ Grafana dashboard
                     └─────────────────────────────┘
```

**Key design decisions:**

| Decision | Why |
|---|---|
| **Namespace isolation** | Two identical questions with a *different* system prompt, model, or temperature must NOT share a cached answer (a "pirate" persona and a "formal" persona shouldn't swap answers). Every cache entry is tagged with a hash of `model + system_prompt + temperature`. |
| **TTL (time-to-live) per entry** | Stable facts ("What is the capital of France?") can be cached for a long time. Time-sensitive questions ("What's the latest news today?") get a short TTL so stale answers don't get served. A simple keyword classifier picks the TTL automatically. |
| **Near-miss logging** | Questions that were *close* to the similarity threshold but didn't quite clear it are logged to `logs/near_misses.jsonl`, so you can later decide whether to loosen the threshold (see `scripts/analyze_near_misses.py`). |
| **OpenAI-compatible request shape** | The proxy accepts the same JSON shape as OpenAI's `/v1/chat/completions` (which Ollama also mirrors), so switching an existing app to use this cache is just a base-URL change. |

---

## 4. Tech Stack — and Why Every Piece Is Free

| Component | Tool | Why It's Free | Why This Choice |
|---|---|---|---|
| Language | Python 3.11+ | Free, open-source | Huge ecosystem, easy to read |
| LLM | [Ollama](https://ollama.com) running a local model (e.g. `llama3.2:1b`) | Free desktop app; models are free to download and run on your own hardware | No API key, no per-token bill, works offline |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Free, open-source model, downloaded once from Hugging Face | Small (~80MB), fast on CPU, good semantic quality |
| Vector store | `ChromaDB` (embedded, file-based) | Free, open-source, no server to install | Zero-config: it just saves files to a folder on disk |
| Proxy layer | `FastAPI` + `uvicorn` | Free, open-source | Drop-in API replacement, automatic docs at `/docs` |
| Cache policy | Custom Python (TTL + similarity threshold) | Free (it's just code) | Fully transparent and tunable |
| Monitoring | `prometheus-client` (built in) + optional Prometheus & Grafana | All free, open-source | Real-time hit-rate/latency dashboards, no cost |
| Containerization | Docker + docker-compose | Docker Desktop is free for personal/small business use | Optional one-command full-stack startup |

No component in this list requires a credit card, a paid tier, or an
account with usage-based billing.

---

## 5. Installation

### Prerequisites (all free)

1. **Python 3.11 or newer** — [python.org/downloads](https://www.python.org/downloads/)
2. **Visual Studio Code** — [code.visualstudio.com](https://code.visualstudio.com/)
   with the **Python extension** (search "Python" in the Extensions panel,
   install the one by Microsoft).
3. **Ollama** — [ollama.com/download](https://ollama.com/download) (free,
   runs LLMs locally on Windows, macOS, or Linux).
4. **Git** — [git-scm.com/downloads](https://git-scm.com/downloads)
   (needed later to push to GitHub).

### Step-by-step setup

**1. Get the project files**

Unzip the project folder anywhere on your computer, then open it in VS Code:

```bash
# From a terminal, inside the unzipped folder:
code .
```

**2. Install and start Ollama, then download a small free model**

After installing Ollama, open a terminal and run:

```bash
ollama pull llama3.2:1b
```

This downloads a small, free, open-source language model (about 1.3GB)
that runs comfortably on most laptops. Ollama starts its own server
automatically in the background (default: `http://localhost:11434`).

> Want a different model? Any model in the [Ollama library](https://ollama.com/library)
> works — just change `DEFAULT_MODEL` in your `.env` file (see step 4) and
> `ollama pull <model-name>` it first.

**3. Create a virtual environment and install dependencies**

In the VS Code integrated terminal (`` Ctrl+` `` / `` Cmd+` ``), from the
project's root folder:

```bash
# Create the virtual environment
python -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat

# Install all dependencies (all free/open-source)
pip install -r requirements.txt
```

In VS Code, once the venv is created, you may be prompted "Select
Interpreter" — choose the one inside `.venv`. This makes sure VS Code's
Run/Debug buttons use the right Python environment.

**4. Set up your configuration file**

```bash
cp .env.example .env
```

The defaults work out of the box — you don't need to edit anything to get
started. Open `.env` in VS Code if you want to tweak settings later (see
[Configuration Reference](#8-configuration-reference)).

**5. Verify everything is wired up correctly**

```bash
pytest
```

You should see all tests pass. (The first run downloads the small
embedding model from Hugging Face — this only happens once.)

---

## 6. Usage

### Start the caching proxy

```bash
uvicorn src.main:app --reload --port 8000
```

Or in VS Code: open the **Run and Debug** panel (left sidebar) and choose
**"Run Semantic Cache API"** — a ready-made launch configuration is
already included in `.vscode/launch.json`.

You should see log output ending in `Semantic cache ready.` Leave this
terminal running.

Visit **http://localhost:8000/docs** in your browser for interactive,
auto-generated API documentation (built into FastAPI, free).

### Try it out

**Option A — the guided demo script** (recommended first step):

```bash
python scripts/demo.py
```

This asks the same question three ways and shows you, in your terminal,
which calls were cache MISSes (went to the LLM) and which were cache
HITs (answered instantly).

**Option B — curl:**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "What is Python used for?"}]
  }'
```

Run the exact same command again — the response comes back with
`"cache_status": "HIT"` and an `X-Cache-Status: HIT` header, near-instantly.

**Check overall statistics:**

```bash
curl http://localhost:8000/stats
```

**Run a realistic load test** (mixed unique and repeated/paraphrased
questions), useful for generating "headline numbers" for a portfolio:

```bash
python scripts/load_test.py --num-requests 200
```

This prints a hit rate, average latency for hits vs. misses, and the
speedup factor, and saves full results to `load_test_results.json`.

**Point your own app at the cache instead of a raw LLM:**
Anything that speaks the OpenAI chat-completions format just needs its
base URL changed to `http://localhost:8000` — no other code changes.

---

## 7. Project Structure

```
semantic-cache-llm/
├── README.md                  ← you are here
├── CONTRIBUTING.md            ← how to contribute changes
├── LICENSE                    ← MIT license (free/open-source)
├── requirements.txt           ← Python dependencies (all free)
├── .env.example                ← copy to .env to configure the app
├── .gitignore                  ← keeps secrets/generated files out of Git
├── Dockerfile                  ← optional: containerize the proxy
├── docker-compose.yml          ← optional: full stack (proxy+Ollama+monitoring)
│
├── src/                        ← application source code
│   ├── main.py                 ← FastAPI app: the proxy's API endpoints
│   ├── config.py               ← all tunable settings, in one place
│   ├── models.py                ← request/response data shapes
│   ├── embeddings.py            ← turns text into meaning-vectors
│   ├── cache_store.py           ← the caching logic (lookup/store/TTL)
│   ├── llm_client.py            ← talks to the free local LLM (Ollama)
│   └── metrics.py               ← Prometheus metrics definitions
│
├── scripts/                    ← standalone helper scripts
│   ├── demo.py                  ← quick guided walkthrough
│   ├── load_test.py             ← realistic traffic + hit-rate report
│   └── analyze_near_misses.py   ← helps you tune the similarity threshold
│
├── tests/                      ← automated tests (pytest)
│   └── test_cache.py
│
├── monitoring/                  ← optional Prometheus + Grafana config
│   ├── prometheus.yml
│   └── grafana/provisioning/…
│
└── .vscode/                    ← VS Code helper configuration
    ├── launch.json              ← one-click Run/Debug buttons
    └── settings.json
```

---

## 8. Configuration Reference

All settings live in `.env` (copied from `.env.example`). Every one has a
sensible free default — you only need to change what you want to
customize.

| Setting | Default | What it controls |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Where your local LLM server is running |
| `DEFAULT_MODEL` | `llama3.2:1b` | Which Ollama model to use |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Which free embedding model to use for meaning-comparison |
| `CHROMA_PERSIST_DIR` | `./cache_data` | Folder where the cache database's files are stored |
| `SIMILARITY_THRESHOLD` | `0.95` | How similar (0–1) two questions must be to count as "the same" — higher = stricter |
| `NEAR_MISS_MARGIN` | `0.10` | How far below the threshold still counts as a "near miss" worth logging |
| `DEFAULT_TTL_SECONDS` | `86400` (24h) | How long a stable-fact answer stays cached |
| `SHORT_TTL_SECONDS` | `3600` (1h) | How long a time-sensitive answer stays cached |
| `ENABLE_METRICS` | `true` | Whether `/metrics` is exposed for Prometheus |

**Tuning the similarity threshold:** Lower it (e.g. `0.90`) to catch more
paraphrases at the risk of occasionally reusing a slightly-wrong answer.
Raise it (e.g. `0.98`) for near-perfect accuracy at the cost of fewer
cache hits. Use `python scripts/analyze_near_misses.py` after some usage
to see real data on where your threshold should sit.

---

## 9. Testing

```bash
pytest
```

This runs `tests/test_cache.py`, which checks (entirely offline, no LLM
required):
- A brand-new question is a cache miss.
- Asking the exact same question again is a cache hit.
- A *reworded* version of the same question is still recognized as similar.
- Expired entries are not served.
- Different system prompts/personas don't leak answers to each other.

---

## 10. Optional: Full Monitoring Stack (Docker)

If you have Docker Desktop installed (free), you can run the entire
stack — the proxy, Ollama, Prometheus, and a pre-built Grafana dashboard
— with one command:

```bash
docker compose up --build
```

Then:
- Proxy API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (login: `admin` / `admin` — change this
  if you expose it beyond your own machine)

The Grafana dashboard ("Semantic Cache Overview") is pre-loaded and shows
cache hit rate, estimated cost saved, cache size, and P50/P95 latency for
hits vs. misses — no manual dashboard setup required.

You don't need Docker at all to use the project day-to-day; this is purely
for a nicer visual dashboard.

---

## 11. Pushing This Project to GitHub

If you're new to Git/GitHub, here's the full path from "unzipped folder"
to "on GitHub," using VS Code's built-in tools.

**1. Create a free GitHub account** at [github.com](https://github.com)
if you don't have one.

**2. Create a new, empty repository on GitHub:**
- Click the **+** icon (top right) → **New repository**
- Name it (e.g. `semantic-cache-llm`)
- Leave it empty — do **not** check "Add a README" (we already have one)
- Click **Create repository** and keep that page open; you'll need the
  URL it shows you (looks like `https://github.com/your-username/semantic-cache-llm.git`)

**3. Initialize Git in your project (VS Code terminal):**

```bash
git init
git add .
git commit -m "Initial commit: semantic caching layer for local LLMs"
```

**4. Connect it to GitHub and push:**

```bash
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/semantic-cache-llm.git
git push -u origin main
```

You'll be prompted to sign in to GitHub the first time — VS Code and Git
will guide you through browser-based authentication.

**Using VS Code's Source Control panel instead of the terminal:**
- Click the **Source Control** icon in the left sidebar (looks like a
  branching line).
- Click **Initialize Repository**.
- Type a commit message at the top, click the **✓ Commit** button.
- Click **Publish Branch** — VS Code will ask you to sign in to GitHub
  and create the remote repository for you automatically.

From now on, after making changes: Source Control panel → write a commit
message → **Commit** → **Sync Changes** (or `git push` in the terminal).

---

## 12. Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the full guide. Short version:
fork it, make a focused change, add/update tests, open a pull request.
Every contribution must stay free to run — no paid APIs or services.

---

## 13. Troubleshooting

| Problem | Likely Cause / Fix |
|---|---|
| `Could not reach Ollama at http://localhost:11434/v1` | Ollama isn't running. Open the Ollama app, or run `ollama serve` in a terminal. |
| First request is very slow | Normal — the embedding model and/or the LLM model are loading into memory for the first time. Subsequent requests are fast. |
| `pip install` fails on `chromadb` or `sentence-transformers` | Make sure you're using Python 3.11+ and that your virtual environment is activated. Try `pip install --upgrade pip` first. |
| Everything is always a cache MISS | Check `SIMILARITY_THRESHOLD` in `.env` — 0.95 is fairly strict. Try lowering it to `0.85` temporarily to confirm the cache logic itself is working, then tune from there. |
| Port 8000 already in use | Run on a different port: `uvicorn src.main:app --port 8001` |
| Want to start fresh / clear the cache | Stop the app, delete the `cache_data/` folder, and restart. |

---

## 14. License

Released under the [MIT License](./LICENSE) — free to use, modify, and
share, including commercially.

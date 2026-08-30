# ModelDebugger

ModelDebugger is a local mechanistic-interpretability interface for Hugging Face causal language-model checkpoints. Its only input is a Hugging Face repository ID and optional revision; it does not accept hand-authored model JSON.

Product purpose, evidence standards, design principles, and the inheritance contract for future components are defined in [`MISSION.md`](MISSION.md).

The circuit diagram and research-paper interface are visually inspired by [Transformer Circuits](https://transformer-circuits.pub/) and its mathematical treatment of residual-stream computation. The interface loads the source site's Styrene A and Tiempos Text webfonts directly from its published CDN, with system fallbacks.

The app reads `config.json` and every Safetensors header without downloading tensor values. The Python backend constructs the transformer’s autoregressive forward function, maps exact checkpoint tensors to operations, preserves unmapped tensors and complete file metadata, and labels evidence-based architecture predictions by source and confidence. The browser receives a ready-to-render graph.

The graph also exposes a residual-stream ledger. It accounts for the embedding, attention write, MLP write, and accumulated residual state at every decoder block, links each row back to the corresponding graph node, and reserves signed activation-norm and direct-logit-attribution fields for prompt-conditioned traces. Because checkpoint metadata does not contain activations, those run-dependent values are explicitly shown as unmeasured instead of being estimated or fabricated.

After an execution worker loads the open checkpoint, **Run model** becomes a prompt-conditioned model interface. A hooked pass displays prompt tokens, next-token probabilities, target rank, output entropy, KV-cache size, execution time and device memory, residual norms, target-token DLA, residual-update norms, convergence to the final residual state, attention/MLP write traces, an attention-head entropy map, and the hook inventory. The same measurements are projected back onto graph nodes and component edges, so charts and circuit topology remain linked. These views are observational and do not claim causal importance.

The debugging workbench persists complete research cases in local SQLite storage. A case records the pinned model/revision, benchmark examples, selected/reference prompts, expected behaviour, tokenizer and chat-template context, metric, seed, dtype/device, generation settings, notes, paired traces, interventions, candidate circuit, and verification results. Saved cases can be reopened after a refresh or server restart from the **Debug cases** library.

The connected workflow includes six behaviour metrics; aligned selected/reference traces; ranked residual, cosine, logit-lens, attention, contribution, and cache divergence; zero/mean/resample/patch/scale/steering interventions; signed causal effects; EAP candidate discovery followed by intervention-backed ACDC pruning; a block/head/token/feature microscope; numerical, hook, cache, latency, and memory diagnostics; JSONL/CSV/Hugging Face benchmark exploration across any outcome; guardrail verification; and a portable Markdown report with settings, evidence tables, a Mermaid circuit graph, results, machine-readable JSON, and caveats.

Circuit discovery is an architecture-neutral, residual-write adaptation inspired by Conmy et al.'s [Towards Automated Circuit Discovery for Mechanistic Interpretability](https://arxiv.org/abs/2304.14997) and Syed, Rager, and Conmy's [Attribution Patching Outperforms Automated Circuit Discovery](https://arxiv.org/abs/2310.10348). It does not vendor the original TransformerLens-specific code: EAP supplies a first-order candidate ranking, then an ACDC-style greedy pass evaluates actual control-activation patches while deleting candidate residual-write edges below the selected metric threshold. The UI reports that bounded adaptation, its fidelity, stability, and limitations rather than presenting it as a reproduction of every edge type in the original implementation.

The application has a public product landing view and a separate research workspace. Launching the workspace opens the checkpoint importer, account session, graph canvas, inspector, residual ledger, and optional Daytona or local execution controls without navigating away from the single-page app.

Tensor ordering is automatic for every inspected repository. Semantic operation order is inferred from checkpoint tensor paths with architecture-neutral aliases and a stable humanized path fallback for unknown modules. Physical order is independently derived from naturally sorted shard names and each tensor's Safetensors byte offset. The UI labels fallback classifications as path-derived, so unfamiliar model implementations remain ordered without pretending their semantic role is known with high confidence.

## Run

```bash
npm run dev
```

The first run creates an isolated `.modeldebugger-app` environment, installs the pinned Daytona SDK, and starts the Python server at <http://localhost:4173>.

For repeatable UX work, use **Open GPT-2 example** below the checkpoint importer. It loads the real `openai-community/gpt2` graph through the normal Hugging Face route and opens a six-example benchmark fixture. Initial row outcomes and scores are explicitly illustrative; running the benchmark with a worker replaces them with observed GPT-2 measurements. The repository and revision fields remain editable for the ordinary any-model workflow.

Use the in-app **Connect** control with a Hugging Face read or fine-grained token to inspect private and gated repositories your account can access. After validation, the token remains only in the loopback server's memory. The browser receives an opaque 30-day session identifier in an HttpOnly, SameSite cookie scoped to the Hugging Face API routes, so refreshes restore the account without exposing the key to page JavaScript or browser-readable storage. Restarting the server invalidates that browser session. Authenticated model responses never enter the shared public-model cache, and **Disconnect** clears both the cookie and the in-memory token. `HF_TOKEN` remains available as an optional server-level credential. Set `PORT` to override the default port.

## Managed Daytona execution

Daytona is the default GPU path. Open **Settings → Execution worker**, paste the Daytona API key for the account that should be billed, review the model-aware GPU recommendation, and choose **Start Daytona GPU**. For private or gated Hugging Face repositories, you can explicitly supply a read token for that private sandbox.

The key is submitted from the masked settings field directly to the loopback backend. After submission, the backend retains it only in memory; it is never placed in browser storage, the SQLite research database, or the sandbox. The backend creates a private spot GPU sandbox—on-demand capacity is never requested—authenticates Daytona's preview gateway and the worker independently, and proxies only fixed execution routes. Spot capacity can be interrupted, so research cases remain stored locally. Model weights and full activation tensors remain in Daytona; the app receives compact summaries and bounded activation slices. Disconnect deletes the sandbox immediately, with a two-hour automatic-deletion backstop to protect limited credits if the local process exits unexpectedly.

The recommendation estimates half-precision weight storage plus framework, KV-cache, and hook-capture headroom. If half precision will not fit on one supported GPU, it recommends 4-bit NF4 loading with BF16 compute. This is a conservative preflight estimate rather than a guarantee for every custom architecture or prompt length. Selecting a smaller override than the estimate is rejected before credits are used.

## Local hooked execution

Choose **Settings → Execution worker → Local machine**, then start a worker in a second terminal:

```bash
make -C '/Users/maximiliannicholson/Documents/untitled folder 5' worker
```

The first run creates `.modeldebugger-worker`, installs PyTorch and the Hugging Face model dependencies, and starts the prefilled `http://127.0.0.1:8765` endpoint. The worker writes its random secret to a user-only `.modeldebugger/local-worker.json` session file, so the loopback backend discovers it automatically without making you copy credentials. The worker selects CUDA or Apple Metal when available and otherwise runs on CPU. It reads private-model credentials from the normal Hugging Face cache or `HF_TOKEN` in the worker process environment.

## Verify

```bash
npm test
```

This compiles the Python package, checks browser graph routing, and runs graph-contract, Safetensors, live HTTPS/Hugging Face, cookie-session, cache, and HTTP-server tests. Validate the Python sources without running tests with:

```bash
npm run build
```

## Prerequisites

- Python 3.11 or newer
- Node.js for the browser routing test

The backend uses Python's standard library for checkpoint inspection and the pinned Daytona Python SDK for managed GPU lifecycle operations. No compiler, libcurl, or cJSON installation is required.

## Backend

The Python backend serves the frontend and exposes:

- `GET /api/health`
- `GET /api/huggingface/account` with a Hugging Face Bearer token or saved session cookie
- `POST /api/huggingface/logout`
- `GET /api/huggingface?model=<repository>&revision=<revision>`
- `GET /api/debug/cases` and `GET /api/debug/cases/<id>`
- `POST /api/debug/cases`, `PUT /api/debug/cases/<id>`, and `DELETE /api/debug/cases/<id>`
- `GET /api/huggingface/dataset?dataset=<repository>` (first 100 rows through the official dataset viewer)
- `GET /api/runtime/status`
- `POST /api/runtime/daytona/recommend`
- `POST /api/runtime/daytona/provision`
- `POST /api/runtime/connect`
- `POST /api/runtime/disconnect`
- `POST /api/runtime/load`
- `POST /api/runtime/forward`
- `POST /api/runtime/compare`
- `POST /api/runtime/intervene`
- `POST /api/runtime/root-cause`
- `POST /api/runtime/verify`
- `POST /api/runtime/activation`

Hugging Face files are inspected with HTTP byte-range requests. The endpoint returns one authoritative `graph` record, avoiding a duplicate legacy payload. It retains each tensor’s raw header entry, dtype, shape, byte offsets, byte size, element count, rank, and shard, plus shard-level `__metadata__`, header and file sizes, ETag, last-modified time, content type, tensor counts, and byte totals. Safetensors does not encode `requires_grad`, so the UI reports exact checkpoint elements and distinguishes recognized buffers from parameter-like storage instead of claiming a trainable-parameter count. The record is fully assembled in Python, including architecture inference, tensor assignment, residual and cache topology, layout coordinates, fallback edge paths, search records, circuit classes, and per-node inspector data. Generation fails closed if its internal validator finds a missing or duplicate tensor, non-conserved element/byte totals, an invalid edge, incomplete layer coverage, or a position/residual topology that contradicts the selected graph contract.

Only browser-native work remains in JavaScript: DOM/SVG creation, obstacle-aware graph routing, viewport virtualization, pan/zoom, keyboard and pointer interaction, accessibility state, forms, loading transitions, token handling, and locale-aware display formatting.

### Project layout

- `refusalscope/__main__.py` — `python -m refusalscope` launcher.
- `refusalscope/server.py` — threaded loopback HTTP server, routing, validation, static serving, auth, sessions, and public response cache.
- `refusalscope/debug_store.py` — thread-safe SQLite persistence for reproducible debugging cases.
- `refusalscope/huggingface.py` — Hub record merging, artifact policy, Safetensors range parsing, tensor inventory, and account sanitization.
- `refusalscope/graph.py` — decoder discovery, architecture evidence, tensor mapping, graph topology, statistics, layout, fallback edge geometry, search records, circuit classes, and inspector projections.
- `refusalscope/http_client.py` — authenticated standard-library HTTPS transport and response capture.
- `refusalscope/config.py` — inspection and request limits.
- `tests/` — unit, graph-contract, live HTTPS/Hub, cookie-session, cache, and server integration tests.
- `src/app.js` — browser rendering and interaction.
- `src/graph-routing.js` — obstacle-aware browser edge routing.
- `src/presentation.js` — locale-aware display formatting that must run in the browser.
- `refusalscope/daytona.py` — model-aware GPU sizing plus private Daytona provisioning and teardown.
- `workers/modeldebugger_worker.py` — authenticated local or managed model and activation worker.

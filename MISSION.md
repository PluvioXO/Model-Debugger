# ModelDebugger Mission

## Purpose

ModelDebugger exists to help researchers debug language models.

It should help a researcher select any benchmark behaviour worth investigating—such as a pass, failure, regression, anomaly, or outlier—see what happened inside the corresponding execution, locate the relevant components and tokens, test a concrete intervention, and verify whether that intervention changes the result. Mechanistic interpretability is a primary diagnostic method for accomplishing this mission, not the mission by itself.

The model should feel like an inspectable running system—not a black-box endpoint, static checkpoint viewer, decorative diagram, or generic inference chat.

## The debugging loop

ModelDebugger should support one connected workflow:

> Reproduce a behaviour → compare it with an expected or control case → trace the internal divergence → localize a candidate cause → intervene → verify the effect → save and share the evidence.

A successful debugging session should help answer:

1. What exactly did the model do?
2. Where in the computation did the relevant behaviour emerge?
3. Which observations support that diagnosis?
4. Does changing the suspected component change the behaviour?
5. Can the result be reproduced by another researcher?

## Product layers

ModelDebugger combines three layers of model debugging:

1. **Reproduction and structure** — pin the model, revision, tokenizer, input, target, and execution settings; reconstruct the forward computation, tensor order, residual topology, attention, MLP, normalization, routing, cache, and output paths.
2. **Observation and localization** — capture a prompt-conditioned execution and connect tokens, predictions, logits, component writes, residual states, attention patterns, cache contents, numerical anomalies, and attribution back to the model graph.
3. **Intervention and verification** — compare a selected benchmark example with a reference case, ablate or patch selected components, steer or replace activations, and measure whether the suspected cause changes the output.

## Core capabilities

Current and future work should strengthen the following capabilities:

- Accept any valid Hugging Face causal-language-model ID and immutable revision without assuming one repository layout, checkpoint format, framework, architecture family, or naming convention.
- Resolve Safetensors, sharded Safetensors, indexed PyTorch checkpoints, monolithic PyTorch checkpoints, configuration-only repositories, adapter-only repositories, and nested text decoders through explicit capability tiers. A missing `model.safetensors` file is a supported resolver state, not an exceptional edge case.
- Produce an explicit capability report for every repository: which files and configuration branches were found, which evidence was used, which model and tensor facts were resolved, which fallbacks were activated, which operations remain uncertain, and which execution or inspection features are unavailable.
- Make every run reproducible by retaining the exact model revision, tokenizer, prompt, tokenization, selected position and target, generation settings, random seed, dtype, device, software versions, and intervention configuration.
- Preserve exact tensor provenance: path, shape, dtype, byte offsets, shard position, physical checkpoint order, and inferred forward-pass role.
- Present transformer blocks, residual-stream states, attention and MLP writes, expert routing, recurrence, caches, and multidimensional tensors as an explorable execution graph.
- Keep graph routes legible: arrows must not cross or pass underneath nodes and tensor stacks; independent signals must retain distinct lanes and connection ports.
- Run models through local workers or private, ephemeral Daytona spot sandboxes while keeping model weights and large activation tensors beside the worker. Managed compute must use spot capacity exclusively, be billed through an API key explicitly supplied by the researcher, recommend capacity from checkpoint evidence, remain private and independently authenticated, and be deleted on disconnect with an automatic expiry backstop.
- Show prompt tokens, output probabilities, logits, entropy, cache statistics, layer traces, attention-head summaries, captured hooks, timing, memory, and numerical failures such as NaN or infinite values.
- Compare any selected benchmark run with a reference or expected run and expose where their outputs and internal states begin to diverge.
- Attribute output support and suppression to captured components using clearly named methods and honest normalization.
- Support causal experiments including zero and mean ablation, activation patching, steering, feature replacement, and clean-versus-corrupted comparisons.
- Measure intervention effects with signed logit and probability changes, rank changes, distribution divergence, and recovery scores where applicable.
- Link every observation and intervention result back to its graph node, edge, tensor, layer, head, token position, target token, and run context.
- Fetch bounded activation slices lazily instead of moving unrestricted activation caches into the browser.
- Retain experiment history and export a portable record that another researcher can rerun and inspect.
- Preserve authenticated Hugging Face identity and worker sessions securely without exposing access tokens to page scripts.

## Hugging Face resolver contract

“Any Hugging Face causal language model” is a compatibility target with measurable behaviour, not a claim that all repositories look like Transformers plus `model.safetensors`.

The resolver must inventory the repository before selecting a strategy. It should recognize nested and composite configuration branches such as `text_config`, `language_config`, decoder-specific configuration, and architecture-specific aliases; tokenizer and generation metadata; shard index files; weight files; adapters; quantization metadata; custom-code declarations; and Hugging Face LFS metadata. File names are hints, not proof of forward semantics.

Resolution should use the strongest available evidence in this order:

1. Pinned repository revision and internally consistent configuration declarations.
2. Checkpoint index manifests and complete, non-executed tensor metadata.
3. Tensor names, shapes, dtypes, shard membership, and physical ordering.
4. Architecture-neutral structural invariants and well-labelled naming aliases.
5. A conservative configuration-only scaffold when weight metadata cannot be inspected safely.

Lower-confidence evidence must never silently override a contradictory higher-confidence source. Every inferred fact should retain its source, confidence, and competing evidence.

The resolver must expose one of four honest capability tiers:

- **Checkpoint mapped** — tensor inventory, shapes, dtypes, storage provenance, and forward-role mapping are complete and internally validated.
- **Manifest mapped** — configuration and checkpoint indexes are available, but some tensor headers, byte offsets, shapes, or dtypes are unavailable. Only supported facts are shown.
- **Configuration scaffold** — the model family and declared dimensions can be represented, but checkpoint tensors cannot be inspected. Nodes and edges derived from configuration are visibly labelled as predicted.
- **Unsupported with diagnosis** — the repository cannot be represented without inventing semantics. The interface reports the exact missing, contradictory, corrupt, unsafe, or unsupported evidence and suggests a concrete recovery path.

Safetensors remains the preferred zero-weight-download inspection path, but it is not a prerequisite for opening a model. When only PyTorch pickle-based weights are present, the loopback inspection service must not execute or unpickle remote checkpoint content merely to draw a graph. It may use a trustworthy index manifest, show a configuration scaffold, or delegate opt-in conversion/introspection to an isolated execution worker. Custom remote code also requires an explicit trust decision and must never run during ordinary metadata inspection.

Repository irregularities are first-class states. The resolver must handle or clearly diagnose:

- monolithic and sharded checkpoints, alternate shard names, subdirectories, and multiple candidate weight sets;
- missing, malformed, stale, or contradictory index manifests;
- Git LFS pointers, truncated files, invalid headers, duplicate tensor names, incomplete shards, and mismatched byte or element totals;
- PyTorch-only, Safetensors-only, mixed-format, quantized, adapter-only, and base-plus-adapter repositories;
- tied weights, absent LM-head tensors, fused or split Q/K/V projections, grouped-query and multi-query attention, mixture-of-experts routing, parallel and sequential residual topologies, recurrent or hybrid blocks, and cache variants;
- multimodal or composite repositories where the causal text decoder is nested beside unrelated encoders;
- private, gated, rate-limited, relocated, revision-deleted, or authentication-dependent artifacts;
- missing standard layer names, unfamiliar tensor prefixes, model-specific buffers, and architectures that require custom code.

Graceful degradation must preserve scientific honesty. The resolver may omit an unsupported operation, retain it as unresolved state, or fall back to a structural scaffold; it may not fabricate tensor provenance, layer coverage, execution order, parameter counts, causal connectivity, or runtime capability. A partial graph must remain useful, visibly partial, and internally consistent.

Compatibility is proven through a maintained resolver matrix, not a handful of familiar models. Tests should cover representative decoder families, nested text decoders, pre-norm and post-norm models, parallel residuals, fused and split projections, GQA/MQA, MoE, tied embeddings, adapters, quantized metadata, unknown architectures, configuration-only repositories, PyTorch-only checkpoints, sharded and malformed manifests, missing files, and authenticated access. Each fixture must assert the selected capability tier, evidence provenance, unresolved facts, validation outcome, and user-facing recovery message. Live probes complement these fixtures but cannot replace deterministic format and failure-mode tests.

## Evidence standard

ModelDebugger must make the strength and meaning of debugging evidence visible.

- **Structural** claims come from pinned configuration, index manifests, non-executed tensor metadata, code-compatible naming patterns, and architecture-neutral invariants. The interface must distinguish checkpoint-mapped facts from manifest-only or configuration-only scaffolds.
- **Observational** measurements come from a specific execution, such as activation norms, attention patterns, logits, cosine similarity, numerical anomalies, or direct logit attribution.
- **Comparative** evidence identifies a difference between explicitly defined runs but does not by itself prove the cause of that difference.
- **Causal** claims require a controlled intervention and a measured effect.
- Predictions and architecture fallbacks must be labelled with their basis and confidence.
- Missing measurements must be shown as unavailable or unmeasured, never estimated for visual completeness.
- Relative attribution shares must state their denominator and must not be described as literal percentages of model causality or output probability.
- Attention weight is not, by itself, component importance.

Every result should retain enough context to be reproduced: model and revision, prompt and tokenization, position and target, execution settings, method, comparison or intervention, and relevant caveats.

## Design philosophy

The interface should be quiet, precise, research-oriented, and organized around the debugging task. It should increase information density without becoming visually noisy.

- Lead with the selected benchmark behaviour, model, and experiment. Authentication, worker setup, and infrastructure belong in Settings.
- Use progressive disclosure. The graph provides orientation; inspectors, comparisons, ledgers, and experiment panels provide depth.
- Make the next debugging action clear: inspect, compare, select a component, intervene, or verify.
- Keep the visual language consistent: restrained paper-like surfaces, muted semantic colours, compact typography, and conventional loaders.
- Transformer-block containers should establish hierarchy without overpowering their contents.
- Multidimensional tensors should read as stacked volumes while preserving the cube glyph that communicates dimensionality.
- Measurements on the graph should use small badges, controlled edge emphasis, and readable legends rather than decorative effects.
- Charts should answer a debugging question, expose scales and signs, and connect back to circuit components.
- Baseline, comparison delta, and intervention effect must remain visually distinct. Positive and negative effects must preserve their sign.
- Empty, loading, disconnected, error, unsupported, out-of-memory, interrupted, and partial-data states are first-class product states.
- Keyboard navigation, accessible names, focus restoration, reduced motion, and responsive layouts are required behaviour.

## Component inheritance contract

Every new component should answer these questions before it is accepted:

1. What model failure or debugging question does it help investigate?
2. What action can the researcher take from the result?
3. Is its evidence structural, observational, comparative, or causal?
4. What model, run, prompt, token position, target, hook, tensor, or intervention gives the value context?
5. How does the user move between this view and the corresponding graph component?
6. Does it work across architectures, and how is fallback or unsupported behaviour labelled?
7. What happens when the data or expected file format is missing, partial, contradictory, unsafe to inspect, too large, numerically invalid, or unavailable?
8. Does it preserve graph legibility, established spacing, typography, colour semantics, and interaction patterns?
9. Can a researcher distinguish a measurement, inference, comparison, and causal effect at a glance?
10. Can the result be reproduced or exported?

Components that cannot answer these questions should be revised before they expand the interface.

## Non-goals

ModelDebugger is not intended to:

- Present model visualization as debugging unless it helps locate or test a concrete problem.
- Imply that a visually prominent or correlated component is causally responsible without intervention evidence.
- Hide model-specific uncertainty behind architecture-specific hardcoding.
- Treat Safetensors, standard Transformers file names, or one `config.json` shape as universal prerequisites.
- Execute remote pickle checkpoints or custom repository code in the metadata-inspection process merely to recover a prettier graph.
- Download full checkpoints merely to draw their structure.
- Transfer unrestricted activation caches into the browser.
- Become a general-purpose chat interface disconnected from internal computation.
- Replace evaluation suites or training systems; it should ingest their examples and help explain passes, failures, regressions, anomalies, and researcher-selected cases.
- Trade diagnostic value, provenance, reproducibility, or honest caveats for visual spectacle.

## Long-term direction

ModelDebugger should become a model-behaviour debugging workbench where architecture, execution, comparison, attribution, intervention, and verification remain connected in one navigable representation. Its success is measured by whether a researcher can move from an unexpected output to a reproducible, evidence-backed diagnosis and a tested change.

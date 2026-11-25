# Implementation notes

## What is preserved

The release keeps the research Triton kernels, native C++ launch helpers, low-rank factor preparation, and cached-output state logic. Local imports are namespaced, internal search paths are removed, and a small public request API replaces the cluster-specific runner. No distributed scheduler, node command client, credential, private dataset, or model weight is included.

Qwen uses the five-kernel parallel-proxy path. InternVL uses its eight-node native-softmax path, including the original bootstrap behavior. The backend directories intentionally remain separate because their native floating-point and launch implementations differ. The public InternVL adapter applies the declared two-threshold rule online. That packaged entry point has not been claimed to reproduce a new full benchmark run.

## Attention and cache semantics

- The full visual and textual KV cache remains present.
- The retained factors are `basis = V_r` and `low_keys = K_V V_r` (equivalently `U_r Sigma_r`). The singular values are absorbed into `low_keys`.
- Each KV head shares its factors with the corresponding query-head group.
- A cached visual output is the conditional visual value readout. It is not a stored full spatial attention distribution.
- Refresh computes exact visual attention. Persist computes exact text attention and approximate visual mass, then combines the updated text contribution with the cached visual output.
- Routing is independent per layer and query head. A proxy-triggered refresh occurs at the current step.
- The research backend has a mandatory initial refresh/bootstrap phase. Its first two cached one-token queries are kept exact while low-rank preparation is started. This convention is preserved.
- The age schedule is `theta * cos(pi/2 * clamp((age - 2)/(D - 2), 0, 1))`. At the horizon, an exact refresh is forced. Age refers to distance from the cached anchor, not global generation position.
- Inter-request factor caches are cleared by the public wrappers. No output or SVD state is reused between samples.

## Supported execution

The fused backends target 36 language layers, 32 query heads, 8 KV heads, and head dimension 128. Supported ranks are 4, 8, 16, 32, 64, and 128. Use BF16, batch size one, one CUDA device, greedy generation, and a dynamic KV cache. Multi-GPU sharding, beam search, padded batches, training, CPU model inference, and concurrent requests within one process are not supported.

The examples provide the model-loading configuration. Model-specific attention implementations are in `vapor._backend.qwen` and `vapor._backend.internvl`.

The public wrapper uses full-key query chunking above 8192 prompt tokens, with blocks of 512 queries. The observer is applied *inside* the query blocks to preserve its attention statistics. This removes the research runner's dataset-specific prefill allowlist. Mathematical attention remains full-key, but long-input memory use, arithmetic ordering, and wall time require GPU comparison before claiming identical paper timing. For shorter prompts, the native eager path is retained.

No new GPU execution has been performed as part of packaging. Source extraction, CPU math checks, import tests, and wheel validation are separate from model-level qualification.

## Statistics

`refresh_rate` is the number of exact-refresh layer/head/step positions divided by refresh plus persist positions counted by the cached decoding backend. It includes mandatory bootstrap queries and excludes the multimodal prefill. It is not a token percentage and is not the same as a sampled state-frequency plot.

`generated_tokens` from the example includes the first generated token, which comes from prefill. The backend's `cached_decode_steps` ordinarily counts the subsequent one-token forwards. Do not silently use generated-token count as its denominator.

The examples intentionally do not advertise latency: first-use compilation and asynchronous preparation must be handled explicitly in any timing experiment.

## Extension points

`VAPORConfig` exposes rank, cache-age horizon, thresholds, and prefill query blocking. The `qwen.vapor` and `internvl.vapor` contexts accept native prepared input IDs. InternVL additionally requires the exact rendered prompt and the fast tokenizer so that user-token offsets can be checked against the generation tokenizer. Both adapters restore model hooks after the request.

The small functions in `vapor.reference` explain and test the tensor math. They are not a substitute for the fused generation backend.

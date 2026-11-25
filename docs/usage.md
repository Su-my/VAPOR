# Usage guide

## Supported examples and validation

The runnable examples cover single-image generation with Qwen3-VL-8B-Instruct and InternVL3.5-8B. Video callers can supply native prepared inputs through the adapters. Benchmark readers and video frame extraction are not bundled.

Package imports, 19 CPU numerical tests, and wheel contents have been checked. GPU smoke testing of the newly packaged generation examples is pending. The results in the README are the paper's reported results, not measurements from these packaging checks.

## Environment

The research environment used Python 3.12, PyTorch 2.8.0, Transformers 4.57.6, Triton 3.4.0, and CUDA 12.8 on A100/A800 GPUs. A C++17 compiler, Ninja, and Python development headers are needed for first-use compilation.

## Example options

- `--model`: a local checkpoint directory or Hugging Face model ID.
- `--revision`: pin a model revision for reproducibility.
- `--local-files-only`: load locally available files without downloading weights.
- `--max-new-tokens`: output limit; the example uses 512, while the paper configuration uses 8192.
- `--rank`: change the proxy rank (default: 16).
- `--fixed-threshold`: disable consistency-based calibration and use a fixed threshold.
- `--output`: save the answer and refresh statistics to a new JSON file.

InternVL uses its native model code via `trust_remote_code=True`. Model downloads and installed dependencies are governed by their respective licenses.

## Defaults

| Setting | Qwen3-VL-8B-Instruct | InternVL3.5-8B |
|---|---:|---:|
| Proxy rank | 16 | 16 |
| Cache-age horizon | 512 | 256 |
| Consistency cutoff | 0.75 | 0.70 |
| Threshold below cutoff | 0.20 | 0.15 |
| Threshold at/above cutoff | 0.10 | 0.05 |
| Maximum output in paper configuration | 8192 | 8192 |

Defaults are also provided in `../configs/`. The example defaults to 512 output tokens for a short trial. Calibration uses the median of valid per-layer/per-head consistency scores over original user-text queries. The extracted observer expects the supported model geometry and at least two eligible prompt tokens per head. It raises an error for unsupported prompts rather than guessing a consistency score. Use a complete question, or explicitly select a fixed threshold.

## Use with a prepared input

The adapters are context managers. They install request-local attention hooks, finish asynchronous preparation, and restore the original hooks even if generation fails. Execute one request per process at a time.

```python
import torch
from vapor import VAPORConfig
from vapor.adapters.qwen import vapor

# model and processor: loaded as in examples/run_image.py
# processor and batch: the model's native multimodal processor output
config = VAPORConfig(max_new_tokens=512)
with torch.inference_mode():
    with vapor(model, processor.tokenizer, batch['input_ids'], config) as request:
        output = model.generate(
            **batch, max_new_tokens=config.max_new_tokens,
            do_sample=False, use_cache=True, cache_implementation='dynamic',
        )
    statistics = request.summary()
```

Pass the same `max_new_tokens` to the adapter and `generate`. Call `generate` once per context. Inputs must contain visual tokens, use batch size one without padding, and fit within 32768 prompt-plus-output tokens. The included examples show model loading and preprocessing. Video callers can supply native prepared batches, but this first release does not include video decoding or benchmark-specific frame readers.

## Layout

```text
src/vapor/
  config.py           Public settings
  adapters/           Model hooks and request lifecycle
  _backend/qwen/      Qwen attention implementation
  _backend/internvl/  InternVL attention implementation
  prefill.py          Query-blocked prefill
  reference.py        Small tensor references for the algorithm's math
examples/             Runnable single-image examples
../configs/              Paper defaults
tests/                CPU numerical and packaging tests
docs/                 Implementation notes
```

Legacy internal names such as `HoldGate` refer to VAPOR's Persist state. They are retained inside the private backend namespace to keep the extracted implementation traceable. Import the public adapters instead of configuring these classes directly.

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The CPU tests do not load weights or require a GPU. They cover full-rank mass recovery, the visual-mass error bound, threshold boundaries, query-chunking equivalence, the scalar state transition, and the backend dependency closure.


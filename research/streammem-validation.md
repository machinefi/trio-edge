# StreamMem Validation Plan

**Status**: Complete — accumulated mode has MRoPE bug; multiframe works; recommend keep experimental (Option C)
**Date**: 2026-03-10

## What StreamMem Does

StreamMem maintains a **bounded KV cache** for continuous video streams. When visual tokens exceed a budget, it scores token importance via proxy-query attention, evicts low-saliency tokens, and merges them into weighted-average "prototype" tokens.

Original paper: [StreamMem: Query-Agnostic KV Cache Memory for Streaming Video Understanding](https://arxiv.org/abs/2508.15717) (Yang et al., 2025)

## Implementation Status

- `streaming_memory.py`: 320 lines, full eviction + prototype merging
- `tests/test_streaming_memory.py`: 340 lines, 15+ test cases
- `examples/bench_streaming_qa.py`: Streaming QA benchmark (synthetic)
- `examples/bench_sink.py` / `bench_sink_real.py`: Attention sink tests
- Config: `TRIO_STREAMING_MEMORY_ENABLED`, `_BUDGET`, `_PROTOTYPE_RATIO`, `_SINK_TOKENS`
- Integration: `generate.py` → `PromptCache` → `StreamingMemory`

## Key Discovery: The OOM Problem Doesn't Exist

### How KV cache actually works (upstream + ours)

Both mlx-vlm and trio-core **create a fresh KV cache per `generate()` call**. There is no cross-call accumulation:

- **mlx-vlm**: `generate_step()` creates `make_prompt_cache()` when `prompt_cache is None`. Cache is a local variable in the generator, GC'd when generator ends.
- **trio-core**: `PromptCacheManager.get_or_create_cache()` calls `c.trim(c.offset)` to reset cache to zero on each call (when StreamMem is OFF).
- **Only when StreamMem is ON** does our code deliberately keep visual KV across calls (trim only decode tokens, not visual).

For video (multiple frames), mlx-vlm concatenates ALL frames into one `pixel_values` tensor and does a **single prefill** — not incremental accumulation. A 100-frame video at 336×336 produces ~7,200 visual tokens in one shot.

### Stress test results (Phase 1)

```
Model: Qwen2.5-VL-3B-Instruct-4bit, 200 synthetic frames

Baseline (StreamMem OFF):
  Memory: 3094MB → 3095MB (+1MB over 200 frames)
  KV offset: 0 → 0 (reset every frame)
  Latency: 1598ms → 188ms (stable after warmup)
  Evictions: 0

StreamMem ON (budget=2000):
  Memory: 3094MB → 3169MB (+76MB)
  Evictions: 187 (every frame after budget exceeded)
  Latency: 253ms → 321ms (70% slower than baseline due to eviction overhead)
```

**Conclusion**: Without StreamMem, memory is flat. The OOM scenario StreamMem claims to solve does not exist in our architecture — because KV cache is rebuilt per call.

## Reframing: What Is StreamMem Actually For?

The OOM story was wrong. The real question is whether **cross-frame KV accumulation provides useful visual memory** — can the model answer questions about previous frames by retaining their visual tokens in the KV cache?

### Concrete scenarios where cross-frame memory matters

**State change detection** — independent frames can only answer "what's here now", accumulated context can answer "what changed":

| Question | Independent frame | Accumulated KV |
|---|---|---|
| "Is there a person?" | Can answer | Can answer |
| "Did someone enter and leave?" | Cannot answer | Could answer |
| "Is this package new or was it always there?" | Cannot answer | Could answer |
| "Are there more cars than 5 minutes ago?" | Cannot answer | Could answer |

**Event sequence understanding** — detecting patterns that span multiple frames:
- Person **loitering** (same person reappearing across frames)
- Vehicle **illegal U-turn** (direction change over time)
- Escalating behavior (gradual approach → sudden movement)

**False positive suppression** — temporal context gives confidence calibration:
- One frame with ambiguous shadow → independent frame may false-alarm
- 10 frames with same static shadow → accumulated model knows it's background

**Other domains**: industrial quality inspection (multi-angle defect detection), surgical monitoring (compare pre/post states), traffic flow analysis, retail analytics (customer dwell time).

### How the original paper validated this

StreamMem paper used 5 benchmarks:

| Benchmark | Type | What it measures |
|---|---|---|
| MLVU | Long video understanding | Holistic/detail/multi-detail subtasks |
| EgoSchema | Very long first-person video | Cross-temporal-span reasoning |
| VideoMME | Multi-modal comprehensive | Various video QA |
| RVS-Ego / RVS-Movie | Streaming video QA | Answer questions during video playback |

**Their core comparison**: different "which tokens to keep" strategies at fixed KV budget (6K tokens):
- Full KV (50K tokens, no compression) = upper bound
- FIFO (keep only recent frames) = lower bound
- StreamMem (saliency selection + prototype merging)

**Key result**: StreamMem at 24K budget **beat full 50K KV** (66.3% vs 65.9% on MLVU) — intelligent eviction removes noise tokens, improving accuracy.

**Critical gap in the paper**: No clean ablation of "accumulated history vs recent-only". They compared compression strategies (FIFO vs saliency), not "with history vs without history". FIFO itself retains some history, so there's no baseline showing that historical tokens are necessary vs just using recent frames.

Also, all their benchmarks input the **complete video first, then ask questions** — not true streaming where questions arrive during the video.

## Our Validation: Cross-Frame Understanding Test

We need a more fundamental experiment. Three modes, compared on the same questions:

| Mode | How it works | What it tests |
|---|---|---|
| **Independent** | Only last frame + question | Baseline — no history |
| **Multiframe** | All frames concatenated as video, one prefill | Upper bound — full context (mlx-vlm native) |
| **Accumulated** | Frames fed one-by-one, KV kept across calls | StreamMem mode — does incremental ≈ batch? |

**If Accumulated ≈ Multiframe >> Independent** → accumulated KV provides real cross-frame memory, StreamMem enables a streaming equivalent of batch video understanding.

**If Accumulated ≈ Independent** → accumulated KV tokens are noise, model doesn't attend to old visual tokens.

### Test cases

Three categories with synthetic frames (controlled ground truth):

**Sequence recall** (3 tests): Show letters/numbers one per frame, ask for the sequence.
- A, B, C → "What letters in order?"
- X, Y, Z → "What letters in order?"
- 1, 2, 3, 4, 5 → "What numbers in order?"

**Disappearance** (3 tests): Show object, then blank frame, ask about it.
- Red circle → blank → "Was there an object earlier?"
- Blue square → blank → "Was there an object earlier?"
- "HELLO" text → blank → "Was there text earlier?"

**Counting** (1 test): Different objects per frame, ask total count.
- Red circle, blue square, green triangle → "How many shapes?"

### Running the experiment

```bash
# All three modes (independent, multiframe, accumulated)
python examples/streammem_crossframe.py

# Specific model
python examples/streammem_crossframe.py -m mlx-community/Qwen2.5-VL-7B-Instruct-4bit
```

## Scripts

| Script | Purpose |
|---|---|
| `examples/streammem_stress.py` | Memory growth stress test (Phase 1 — done) |
| `examples/streammem_crossframe.py` | Cross-frame understanding validation (Phase 2) |
| `examples/bench_streaming_qa.py` | POPE accuracy after streaming warmup (existing) |

## Decision Framework (updated)

| Result | Action |
|---|---|
| Accumulated ≈ Independent (no cross-frame memory) | **Remove StreamMem** — accumulated KV is just noise |
| Accumulated ≈ Multiframe >> Independent | **Keep StreamMem** — it enables streaming video understanding |
| Accumulated > Independent but << Multiframe | **Keep as experimental** — partial value, needs improvement |
| Accumulated works but eviction degrades it | **Rework eviction** — the accumulation idea is sound but implementation needs tuning |

### Verdict: Accumulated mode has a position ID bug — needs fix before conclusions

The cross-frame test results **do not support the original "StreamMem is useless" conclusion**. What we actually learned:

1. **Multiframe (batch) video understanding works.** The model correctly recalls content across frames. Our test prompts were poorly designed.
2. **Accumulated KV mode is broken due to MRoPE position ID conflict.** Each `analyze_video()` call recomputes position IDs from 0, but KV cache is concatenated → attention computes on scrambled positions. This is an implementation bug.
3. **We cannot evaluate StreamMem's value until accumulated mode works correctly.**

### What needs to happen next

**Option A — Fix position IDs in accumulated mode:**
- When appending new frame's KV, continue MRoPE position IDs from where the previous frame ended
- This requires: (1) saving the last position ID state after each frame, (2) passing it as the starting point for the next frame's `get_rope_index()` call
- Engineering effort: medium (MRoPE has 3 dimensions: temporal, height, width)
- If this works, re-run cross-frame tests → if Accumulated ≈ Multiframe, StreamMem is validated

**Option B — Reframe StreamMem as "streaming multiframe":**
- Instead of accumulating KV across independent calls, implement proper incremental video encoding
- Each new frame is appended to the existing video context with correct temporal position
- This is closer to what the StreamMem paper actually does
- Engineering effort: high

**Option C — Keep status quo:**
- StreamMem stays experimental, default OFF
- Multiframe (batch) already works for video understanding
- Watch mode (per-frame yes/no) doesn't need history
- Revisit when a concrete product need for streaming understanding emerges

**Current recommendation: Option C.** The cross-frame understanding capability exists in multiframe mode. No product feature currently requires streaming accumulation. The position ID fix (Option A) is worth doing only if a concrete use case demands it.

## Results

### Phase 1: Memory stress test (complete)

Baseline (no StreamMem) shows **zero memory growth** — KV cache rebuilt per call. OOM is not a real problem.

### Phase 2: Cross-frame understanding (complete)

**Qwen2.5-VL-3B** (4-bit):

| Mode | Sequence (3) | Disappearance (3) | Counting (1) | Total |
|---|---|---|---|---|
| Independent (last frame only) | 1/3 | 0/3 | 0/1 | **1/7 (14%)** |
| Multiframe (all frames, one prefill) | 0/3 | 1/3 | 0/1 | **1/7 (14%)** |
| Accumulated (KV kept across calls) | 0/3 | 0/3 | 0/1 | **0/7 (0%)** |

**Qwen2.5-VL-7B** (4-bit):

| Mode | Sequence (3) | Disappearance (3) | Counting (1) | Total |
|---|---|---|---|---|
| Independent (last frame only) | 1/3 | 0/3 | 0/1 | **1/7 (14%)** |
| Multiframe (all frames, one prefill) | 0/3 | 2/3 | 0/1 | **2/7 (29%)** |

### Phase 2b: Diagnostic — what actually went wrong? (complete)

The 14% multiframe result looked like a model capability issue, but **further testing disproved this**.

**Single-frame sanity check**: 3B model correctly identifies ALL synthetic frames individually:
- Letters A, B, C, X → all correct
- Red circle, blue square, green triangle → all correct
- Blank frame → correctly identified

**Multiframe diagnostic** (3 frames: A, B, C as video):

| Question | Answer | Correct? |
|---|---|---|
| "List all letters you see" | A, B, C | **Yes** |
| "What letter in the first frame?" | B | No (temporal order confused by patching) |
| "How many frames?" | 2 | No (temporal_patch_size=2 merges adjacent frames) |
| "Does letter A appear? yes/no" | No | **No** — yes/no format on video always returns No (3B quirk) |
| "Does letter B appear? yes/no" | No | Same quirk |

**Disappearance diagnostic** (red circle → blank, as 4-frame video):

| Question | Answer | Correct? |
|---|---|---|
| "What was the object that disappeared?" | Red circle | **Yes** |
| "Was there a red circle?" | Yes | **Yes** |
| "Was there a blue square?" | No | **Yes** |

### Root cause analysis

**The model CAN do cross-frame reasoning.** Multiframe mode works correctly — the model sees all frames and can recall their content. The original 14% result was caused by:

1. **Prompt design mismatch**: Asking "list letters separated by commas" with strict substring matching. The model's natural response format didn't always match our ground truth parser.
2. **Yes/no quirk**: 3B model in video mode answers "No" to all "does X appear?" questions — a model quirk, not a capability gap.
3. **Temporal order confusion**: Qwen2.5-VL's `temporal_patch_size=2` merges adjacent frames into temporal patches. The model can recall WHAT it saw but not necessarily WHICH FRAME it was in.

**Accumulated mode (0%) fails for a completely different reason**: MRoPE position ID conflict. Each `analyze_video()` call computes position IDs starting from 0, but the KV cache is concatenated. So frame 2's position IDs overlap with frame 1's, causing attention to compute on scrambled positions. This is an **implementation bug**, not a fundamental limitation of KV accumulation.

### Corrected interpretation

| Original conclusion | Corrected conclusion |
|---|---|
| "Model can't do cross-frame reasoning" | **Wrong.** Model does cross-frame reasoning correctly in multiframe mode. Test prompts were bad. |
| "Accumulated KV is harmful — the idea doesn't work" | **Partially right, wrong reason.** Accumulated KV fails because of MRoPE position ID conflict, not because accumulation is fundamentally flawed. |
| "StreamMem provides no value" | **Premature.** We haven't tested accumulated KV with correct position IDs yet. |

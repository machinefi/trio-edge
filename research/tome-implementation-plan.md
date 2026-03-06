# ToMe Implementation Plan for trio-core

## Goal

Implement Token Merging (bipartite soft matching) inside Qwen2.5-VL's vision encoder,
as a training-free visual token compression layer. First VLM inference engine to do this.

## Architecture Overview

Current Qwen2.5-VL vision encoder pipeline:

```
pixel_values
    ↓
PatchEmbed (Conv3d) → (seq_len, hidden_dim)     # 1176-dim patches
    ↓
[rotary_pos_emb + window_index computation]
    ↓
┌─ Qwen2VLVisionBlock 0  (windowed attention)
├─ Qwen2VLVisionBlock 1  (windowed attention)
├─ ...
├─ Qwen2VLVisionBlock 7  (FULL attention)        # fullatt_block_indexes
├─ ...
├─ Qwen2VLVisionBlock 15 (FULL attention)
├─ ...
├─ Qwen2VLVisionBlock 23 (FULL attention)
├─ ...
└─ Qwen2VLVisionBlock 31 (FULL attention)
    ↓
PatchMerger (2×2 spatial merge, MLP) → (seq_len/4, out_hidden_dim)
    ↓
reverse window_index ordering
    ↓
hidden_states → passed to LLM
```

Key file: `.venv/lib/.../mlx_vlm/models/qwen2_5_vl/vision.py`

**We do NOT modify mlx_vlm source.** Instead, we monkey-patch or wrap the VisionModel
to inject ToMe merging between blocks.

## Step-by-Step Implementation

### Step 1: Core ToMe Algorithm (`src/trio_core/tome.py`)

Implement bipartite soft matching as described in paper Appendix D.

```python
def bipartite_soft_matching(metric: mx.array, r: int) -> tuple[merge_fn, unmerge_fn]:
    """
    Args:
        metric: (N, D) token features to compute similarity (use K from attention)
        r: number of tokens to merge this step

    Returns:
        merge: function that merges tokens (N, D) → (N-r, D)
        unmerge: function to reverse (for skip connections if needed)

    Algorithm:
        1. Split tokens into sets A (even indices) and B (odd indices)
        2. Compute cosine similarity between A and B
        3. For each token in A, find most similar in B
        4. Keep top-r most similar pairs
        5. Merge by weighted average
    """
```

Key details:
- Use cosine similarity on the **Key (K) matrix** from attention, not raw features
- Weighted average preserves information (weight by token `size`)
- Track token `size` for proportional attention correction
- All ops must be in MLX (`mx.array`) for GPU acceleration

### Step 2: Wrap Vision Encoder (`src/trio_core/tome_vision.py`)

Create a wrapper that injects ToMe into the existing VisionModel without forking mlx_vlm.

```python
class ToMeVisionWrapper:
    """Wraps Qwen2.5-VL VisionModel to add token merging."""

    def __init__(self, vision_model, r: int = 8, ratio: float = None):
        self.vision_model = vision_model
        self.r = r  # tokens to merge per layer (constant schedule)
        # or
        self.ratio = ratio  # fraction of tokens to merge total

    def __call__(self, hidden_states, grid_thw, **kwargs):
        # Replicate VisionModel.__call__ but with ToMe between blocks
        ...
```

**Approach A: Monkey-patch the block loop**
- Replace `vision_model.__call__` with our version that adds merge steps
- Pro: minimal code, no duplication
- Con: fragile if mlx_vlm updates

**Approach B: Wrap and delegate**
- Copy the 30-line `__call__` method, add merge steps
- Pro: explicit, no monkey-patching
- Con: need to update if mlx_vlm changes
- **Prefer this approach** — the method is short and stable

### Step 3: Handle Windowed Attention Interaction

Qwen2.5-VL uses **windowed attention** for most ViT blocks (only blocks at
`fullatt_block_indexes` use full attention). This means:

- Tokens are reordered by `window_index` before the block loop
- `cu_seqlens` tracks window boundaries
- **ToMe must only merge tokens within the same window**

Solution: use `cu_window_seqlens` to identify window boundaries, run bipartite
matching independently per window.

```python
def tome_merge_windowed(hidden_states, cu_seqlens, r_per_window):
    """Merge tokens respecting window boundaries."""
    merged_segments = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i+1]
        window_tokens = hidden_states[start:end]
        merged = bipartite_merge(window_tokens, r=r_per_window)
        merged_segments.append(merged)
    return mx.concatenate(merged_segments, axis=0), new_cu_seqlens
```

### Step 4: Handle rotary_pos_emb After Merging

After merging tokens, the positional embeddings must be updated to match.

Options:
- **Option A:** Merge pos_emb with same indices as token merge (average positions)
- **Option B:** Re-index remaining positions sequentially
- **Option C:** Drop pos_emb for merged tokens (the ViT continues from attention output)

Paper uses Option A (merge pos embeddings alongside tokens). Since ToMe is between
attention and MLP, and the next block will compute fresh attention with updated tokens,
the position information propagates naturally.

### Step 5: Handle PatchMerger Interaction

PatchMerger expects input shaped as `(seq_len, hidden_dim)` where `seq_len` is a
multiple of `spatial_merge_unit` (= spatial_merge_size² = 4).

After ToMe, `seq_len` is reduced and may not be a multiple of 4. Options:
- **Option A:** Pad to multiple of 4 before PatchMerger (simple, small waste)
- **Option B:** Apply ToMe only up to a layer before PatchMerger, keeping exact multiples
- **Option C:** Adjust r per layer to maintain divisibility

Start with Option A (simplest), optimize later.

### Step 6: Integrate into CompressedMLXBackend

```python
class CompressedMLXBackend(MLXBackend):
    def load(self):
        super().load()
        # Wrap the vision model with ToMe
        self._model.vision_tower = ToMeVisionWrapper(
            self._model.vision_tower,
            r=self.tome_r,
        )
```

After wrapping, the rest of the pipeline (get_input_embeddings, generate) works
unchanged — the vision tower just outputs fewer tokens.

**This is the cleanest integration point** — no need for a custom generate loop.
The LLM side sees fewer visual tokens naturally.

### Step 7: Update Eval Framework

Add to eval suite:
- Vision encoder time (separate from LLM prefill)
- Token count before/after ToMe
- ToMe overhead (time spent on matching/merging)
- Per-layer merge statistics

### Step 8: Benchmark at Scale

Test matrix:

| Resolution | Frames | ~Visual Tokens | Expected ToMe Speedup |
|-----------|--------|---------------|----------------------|
| 480×640   | 2      | 419           | 1.3-1.5×             |
| 720×1280  | 2      | ~1600         | 1.5-2×               |
| 1080×1920 | 2      | ~3600         | 2-3×                 |
| 720×1280  | 8      | ~6400         | 2-4×                 |
| 1080×1920 | 8      | ~14400        | 3-5×                 |

## Key Decisions

### What r value to use?

Paper uses constant r (merge r tokens per layer). For Qwen2.5-VL with 32 blocks:
- Conservative: r=4 → removes 128 tokens total (~30% of 419)
- Moderate: r=8 → removes 256 tokens total (~61% of 419)
- Aggressive: r=12 → removes 384 tokens total (~92% of 419)

**Start with r=8** and sweep r={2,4,8,12,16} in eval.

For large inputs (6000+ tokens), can use higher r since there's more redundancy.
Consider making r proportional to input token count.

### Constant vs decreasing schedule?

Paper finds constant schedule is near-optimal (Figure 2). Decreasing schedule
(more merging early, less later) is slightly faster for same quality.

**Start with constant, add decreasing as an option.**

### Where NOT to merge?

- **Full attention layers** (fullatt_block_indexes): these are critical for global
  information flow. Merging here may hurt quality disproportionately.
  → Start by only merging in windowed attention layers, skip full attention layers.

- **First and last blocks**: first block has raw patch features (not yet semantically
  meaningful), last block feeds into PatchMerger.
  → Skip first 2 and last 2 blocks initially.

## File Structure

```
src/trio_core/
├── tome.py                    # Core bipartite soft matching algorithm
├── tome_vision.py             # VisionModel wrapper with ToMe injection
├── compressed_backend.py      # Updated: use ToMe wrapper instead of post-encoder compression
├── token_compression.py       # Keep: post-encoder compression as fallback/comparison
└── eval.py                    # Updated: vision encoder timing, token count tracking

research/
├── README.md
├── visual-token-compression.md
├── tome-implementation-plan.md  # This file
└── eval-results/
    ├── eval_baseline.json
    ├── eval_compressed_50.json
    └── eval_compressed_75.json
```

## Qwen ViT Architecture Differences

Qwen2.5-VL and Qwen3-VL have **different** vision encoders. The ToMe wrapper must be model-specific.

| Feature | Qwen2.5-VL | Qwen3-VL |
|---------|-----------|----------|
| Position Embedding | 3D RoPE (rotary) | `nn.Embedding` + bilinear interpolation (absolute) |
| Attention | Windowed + full attention (`fullatt_block_indexes`) | All full attention, no windows |
| Multi-scale features | None | `deepstack_visual_indexes` extracts from intermediate layers |
| Output | `hidden_states` | `(hidden_states, deepstack_feature_lists)` tuple |
| Block type | `Qwen2VLVisionBlock` | `Qwen3VLMoEVisionBlock` |

**Implications for ToMe:**
- **Qwen2.5-VL** (harder): Must merge within windows, handle `cu_seqlens`/`window_index`, skip `fullatt_block_indexes`
- **Qwen3-VL** (easier): No windows, can merge across entire sequence freely
- **Strategy**: Keep `bipartite_soft_matching()` model-agnostic, put model-specific logic in separate wrappers
- **Priority**: Implement Qwen2.5-VL first (current default model), then Qwen3-VL

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Windowed attention + ToMe interaction breaks quality | High | Merge only within windows, skip full-attn layers |
| PatchMerger shape mismatch after ToMe | Medium | Pad to multiple of 4, or adjust r |
| mlx_vlm update breaks our wrapper | Medium | Pin mlx_vlm version, keep wrapper minimal |
| Qwen3-VL has different ViT architecture | Medium | Separate wrappers per model family, shared core algorithm |
| Merging overhead exceeds savings at small token counts | Low | Make ToMe optional, auto-disable below threshold |
| Position embedding mismatch after merge | Medium | Merge pos embeddings alongside tokens |

## Success Criteria

1. **Prefill latency**: ≥30% reduction at 480×640, ≥50% at 1080p
2. **Quality**: <1% degradation on descriptive prompts (manual eval)
3. **Vision encoder time**: ≥30% reduction
4. **Overhead**: ToMe matching adds <5% to total vision encoder time
5. **No training required**: works off-the-shelf with any Qwen2.5-VL checkpoint

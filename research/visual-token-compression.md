# Visual Token Compression for Edge VLM Inference

## Problem Statement

In video VLM inference, visual tokens dominate the input sequence. For Qwen2.5-VL-3B:
- A single 480×640 2-frame input produces 419 visual tokens out of ~450 total input tokens
- At 1080p with 8 frames, visual tokens scale to ~6000+
- Prefill cost is O(n²) due to self-attention — **doubling visual tokens quadruples prefill time**
- Each visual token also creates a KV cache entry that persists through the entire generation

Yet visual tokens have massive redundancy: a surveillance frame that's 70% wall produces
the same number of tokens as a complex street scene. The model doesn't adapt its token count
to visual complexity.

## Three Key Papers

### 1. Token Merging (ToMe) — ICLR 2023 (Oral)

**Paper:** Bolya et al., "Token Merging: Your ViT But Faster" ([arXiv:2210.09461](https://arxiv.org/abs/2210.09461))
**Code:** [github.com/facebookresearch/ToMe](https://github.com/facebookresearch/ToMe)

**Core Idea:** Gradually merge similar tokens *inside* the Vision Transformer, between each
block's attention and MLP layers. No training required.

**Algorithm — Bipartite Soft Matching:**
1. Partition tokens into two sets A and B (alternating assignment)
2. Compute cosine similarity between A and B using attention **Keys** (K matrix)
3. Each token in A draws one edge to its most similar token in B
4. Keep the top r most similar edges
5. Merge connected tokens (weighted average by token size)
6. Concatenate A and B back together

**Key Design Choices (Table 1 ablations):**
- Use K matrix for similarity (not raw features X) — K summarizes token information
- Cosine similarity > euclidean > dot product
- Weighted average > keep-one > max-pool > avg-pool
- Alternating partition > sequential > random
- Proportional attention: A = softmax(QK^T/√d + log s) where s = token size

**Results:**
- ViT-L/16 @ 512px: **2× throughput, only 0.2-0.3% accuracy drop**
- Removes 96-98% of tokens over the full network
- Works on images, video, and audio without modification
- Merging overhead is "absolutely negligible"

| Method | Accuracy | Speed (im/s) |
|--------|----------|---------------|
| Baseline (no merging) | 85.96 | 93.3 |
| Random prune | 79.22 | 184.4 |
| K-means merge | 80.29 | 147.5 |
| **Bipartite matching (ToMe)** | **84.25** | **182.9** |

**Why it matters for trio-core:**
- **Training-free** — preserves Qwen's generality. No domain-specific fine-tuning needed.
- Compresses **inside** the vision encoder — speeds up the encoder itself, not just LLM prefill.
- Proven on video (Section 5) — 2.2× throughput on video ViTs with 0.2% accuracy drop.
- Simple to implement — the bipartite matching is ~20 lines of code (Appendix D).

### 2. Visual Context Compression — NeurIPS 2024

**Paper:** "Efficient Large Multi-modal Models via Visual Context Compression"

**Core Idea:** Train a lightweight compression module (few MLP layers + attention pooling)
between the vision encoder output and the LLM input. Compresses visual token embeddings
before they enter the language model.

**Approach:**
- Insert a learned compression layer after vision encoder, before LLM
- Train this layer (with frozen encoder + LLM) on visual instruction data
- The layer learns which tokens are redundant and how to aggregate them

**Relevance to trio-core:**
- More aggressive compression possible (up to 75-90% reduction with <1% quality loss)
- **But requires training** — compression layer is fit to training data distribution
- Loses generality: a compression layer trained on indoor scenes may not work for driving
- Better suited as a **second stage** optimization for specific deployment scenarios

### 3. Inference Optimal VLMs — ICLR 2025

**Paper:** "Inference Optimal VLMs Need Fewer Visual Tokens and More Parameters"

**Core Idea:** Under a fixed inference budget (latency or memory), the optimal VLM design
should use fewer visual tokens and allocate more parameters to the language model.

**Key Insight:**
- Current VLMs over-invest in visual tokens (high resolution, many frames)
- Better to reduce visual tokens and use a larger/better LLM
- This is an architectural design principle, not a runtime technique

**Relevance to trio-core:**
- Validates our direction: reducing visual tokens is the right lever to pull
- Provides theoretical backing for why compression works
- Suggests that model selection (e.g., 7B LLM with fewer visual tokens vs 3B with many)
  is also an important knob

## Our Prototype Results

### Setup
- Model: Qwen2.5-VL-3B-Instruct-4bit on Apple M3 Pro
- Input: 480×640, 2 frames (synthetic test images)
- Dedup/motion gate disabled for clean measurement
- Compression applied post-encoder (between vision_tower output and LLM)

### Baseline (no compression)

| Metric | Value |
|--------|-------|
| prompt_tokens | 419 |
| prefill_ms | ~1016ms |
| decode_ms | ~1030ms (64 tokens) |
| inference_ms | ~2080ms |
| prompt_tps | ~412 tok/s |
| generation_tps | ~62 tok/s |
| peak_memory | 3.82GB |

**Key observation:** All complexity levels (solid, gradient, scene, noise) produce exactly
419 prompt tokens. The model wastes tokens on simple scenes.

### 50% Uniform Compression (keep every 2nd token)

| Metric | Baseline | Compressed | Change |
|--------|----------|-----------|--------|
| prompt_tokens | 419 | 223 | **-46.8%** |
| prefill_ms | 1016 | 280 | **-72.5%** |
| decode_ms | 1030 | 1283 | +24.5%* |
| inference_ms | 2080 | 2075 | ≈ flat |
| peak_memory | 3.82GB | 3.83GB | ≈ flat |
| quality | Normal | Usable but degraded | — |

*Decode increase is because model generates longer responses, not slower per-token.

### Analysis

1. **Prefill scales sub-linearly with token reduction** — 47% fewer tokens → 72% faster prefill
   (expected: attention is O(n²), so ~75% reduction. We measured ~72%, consistent.)

2. **Total latency didn't improve** — because 419 tokens is small. Prefill is only ~49% of
   total inference time. With larger inputs (1080p, 8+ frames, ~6000 tokens), prefill would
   dominate and compression would show massive end-to-end gains.

3. **Quality is the bottleneck** — our naive uniform stride loses too much information.
   ToMe's bipartite matching on K-matrices would retain quality while being just as fast.

4. **Peak memory unchanged** — at 419 tokens, KV cache is not the bottleneck. At 6000+ tokens
   with longer generations, KV cache reduction from compression would be meaningful.

## Compression vs KV Cache: How They Interact

Visual token compression and KV cache optimization (e.g., StreamingLLM, Phase 2 VideoCache)
are **complementary, not conflicting**:

```
Input tokens → [Compression] → fewer visual tokens → LLM prefill → KV cache → Decode
                ^^^^^^^^^^^^^^^                                      ^^^^^^^^^^
                ToMe reduces this                                    StreamingLLM manages this
```

**How compression helps KV cache:**
- Fewer visual tokens → fewer KV entries created during prefill
- Each generation step's attention over KV cache is faster (O(n) per step)
- Total KV memory reduced proportionally to token reduction
- More headroom for longer generation or larger context windows

**Interaction with StreamingLLM:**
- StreamingLLM keeps "attention sink" tokens (usually first few) + sliding window
- With compression, visual tokens are fewer but each carries more information
- Need to mark compressed visual tokens as "important" during KV eviction
- Token size metadata (from ToMe's proportional attention) can guide eviction priority

**Interaction with VideoCache (Phase 2):**
- VideoCache reuses KV entries across similar frames
- Compressed tokens are more semantically meaningful (redundancy removed)
- Cache hit rate should *improve* because compressed representations are more stable
- Fewer tokens per frame → more frames fit in a fixed cache budget

**Net effect: compression + KV cache optimization = compound improvement:**

| Scenario | Prefill | KV memory | Decode speed |
|----------|---------|-----------|-------------|
| Baseline | 1× | 1× | 1× |
| Compression only (ToMe) | 4-10× faster | 0.25× | 1× |
| KV cache only (StreamingLLM) | 1× | sliding window | 1.2-1.5× |
| **Both combined** | **4-10× faster** | **0.1-0.25×** | **1.2-1.5×** |

## ToMe Adoption in Inference Engines

### Where ToMe IS adopted
- **Stable Diffusion** — `tomesd` library, widely used in community for 2× faster image gen
- **ViT classification** — original paper benchmarks, academic adoption
- **HuggingFace Diffusers** — ToMe integration for diffusion models

### Where ToMe is NOT adopted (as of March 2026)
- **vLLM** — no visual token merging
- **mlx-vlm** — no token merging in vision encoder
- **SGLang** — no vision-side optimization
- **TensorRT-LLM** — quantization/batching focus, no ViT token merging
- **Roboflow Inference** — no token merging
- **Any VLM inference engine** — **nobody is doing this**

### Why the gap exists
1. VLM inference engines focus on **LLM-side** optimization (KV cache, batching, quantization)
2. Vision encoder is treated as a black box that runs once per request
3. For cloud deployment with batching, prefill is amortized — less motivation to optimize
4. For **edge/real-time** deployment (trio-core's target), prefill per-request matters enormously

### Opportunity for trio-core
This is a genuine differentiation opportunity. If trio-core implements ToMe in the Qwen2.5-VL
vision encoder:
- First VLM inference engine with training-free visual token compression
- Direct impact on the metric that matters most for edge: per-request latency
- Compounds with existing temporal dedup / motion gating (frame-level → token-level)
- The full pipeline becomes: skip frames (motion gate) → skip tokens (ToMe) → skip KV (cache)

## Implementation Plan

### Phase 1: ToMe in Vision Encoder (training-free) ← **DO THIS FIRST**

Insert bipartite soft matching into `VisionModel.__call__()` in Qwen2.5-VL:

```python
# Current vision encoder loop (qwen2_5_vl/vision.py line 379-387):
for layer_num, blk in enumerate(self.blocks):
    hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, rotary_pos_emb=rotary_pos_emb)

# With ToMe:
for layer_num, blk in enumerate(self.blocks):
    hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, rotary_pos_emb=rotary_pos_emb)
    if layer_num not in self.fullatt_block_indexes:  # Don't merge in full-attention layers
        hidden_states = tome_merge(hidden_states, r=r_per_layer)
```

Implementation details:
- Extract K matrix from the block's attention for similarity computation
- Bipartite matching is ~20 lines (see ToMe Appendix D)
- Need to update `cu_seqlens` and `rotary_pos_emb` after each merge
- `r` (tokens to merge per layer) is the main hyperparameter
- With Qwen2.5-VL's 32 ViT blocks, r=8 per layer removes 256 tokens total

**Estimated effort:** 1-2 days
**Expected impact:** 1.5-2× vision encoder speedup, 2-4× prefill speedup, <0.5% quality drop

### Phase 2: Eval at Scale

Run eval suite at higher resolutions and frame counts where compression matters:
- 1080p single frame (~2400 visual tokens)
- 720p 8 frames (~6000 visual tokens)
- 1080p 16 frames (~19000 visual tokens)

This will show the quadratic speedup benefit clearly.

### Phase 3: Integration with KV Cache

Combine ToMe with Phase 2 VideoCache:
- Use ToMe token size metadata to guide KV eviction priority
- Benchmark compound effect of compression + cache
- Target: real-time webcam analysis at 1080p

### Phase 4 (Optional): Learned Compression for Specific Deployments

For customers with specific use cases (surveillance, retail, manufacturing):
- Train a lightweight compression layer on domain data
- Can push compression further (75-90% reduction)
- Trade generality for maximum speed in known scenarios

## Open Questions

1. **Qwen2.5-VL uses windowed attention in the ViT** — how does ToMe interact with
   window boundaries? May need to only merge within the same window.

2. **PatchMerger already does 2×2 spatial merging** — ToMe sits before this. Does the
   combination of ToMe + PatchMerger cause issues?

3. **Temporal dimension** — for video frames, should we merge across time (different frames)
   or only within the same frame? ToMe paper Section 5 suggests cross-frame merging works.

4. **Proportional attention in Qwen's LLM** — Qwen2.5-VL's LLM doesn't have proportional
   attention. After ToMe merging, should we add the log(s) correction to the LLM's attention
   as well, or is it only needed inside the ViT?

## References

1. Bolya et al., "Token Merging: Your ViT But Faster", ICLR 2023
   https://arxiv.org/abs/2210.09461
2. "Efficient Large Multi-modal Models via Visual Context Compression", NeurIPS 2024
3. "Inference Optimal VLMs Need Fewer Visual Tokens and More Parameters", ICLR 2025

## Compatibility with Future Optimizations

ToMe operates at the **input reduction** layer. All other inference optimizations operate at the
**compute/memory** layer. They are orthogonal and compound.

```
Frame-level:  Motion Gate → Temporal Dedup → fewer frames
                                                  ↓
Token-level:  ToMe (inside ViT) ──────────→ fewer visual tokens
                                                  ↓
Compute-level: Quantization (INT4) ────────→ cheaper per-token compute
                                                  ↓
Memory-level:  KV Cache / StreamingLLM ───→ bounded memory during decode
                                                  ↓
Throughput:    Batch Scheduling ───────────→ more concurrent requests
```

**KV Cache / StreamingLLM:**
- Fewer visual tokens → fewer KV entries created at prefill → less to manage
- Compressed tokens carry more info per token → guide eviction (don't evict merged tokens)
- Token `size` metadata from ToMe can be stored alongside KV entries for priority
- Net: compression makes KV cache optimization easier, not harder

**Batch Scheduling:**
- Shorter sequences → less memory per request → larger batch sizes possible
- ToMe produces fixed-length reduction (r tokens per layer) → predictable sequence lengths
- Predictable lengths are better for batch padding efficiency
- Net: purely positive, no interaction issues

**Quantization (INT4/INT8):**
- ToMe works in the floating-point embedding space of the ViT
- Quantization compresses model weights, not token counts
- One reduces "how many tokens", the other reduces "cost per token"
- Can be applied simultaneously without conflict
- Net: multiplicative speedup (fewer tokens × cheaper per token)

**Summary: no conflicts. Every future optimization we implement will benefit from
having fewer tokens to work with. ToMe is a foundation layer that amplifies everything above it.**

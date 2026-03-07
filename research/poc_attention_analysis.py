"""PoC: Analyze text→visual attention distribution in Qwen2.5-VL.

Goal: Determine if visual token attention is long-tailed (few tokens get
most attention → pruning viable) or flat (all tokens equally important →
pruning will hurt).

Method: After prefill, manually compute Q·K^T attention scores at each
layer using the cached KV states and the hidden states.
"""

import numpy as np
import mlx.core as mx
from PIL import Image

from mlx_vlm import load


def main():
    model_name = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
    print(f"Loading {model_name}...")
    model, processor = load(model_name)

    # Create test image with visual variety
    img = Image.new("RGB", (448, 448))
    pixels = img.load()
    for x in range(448):
        for y in range(448):
            pixels[x, y] = (
                int(100 + 100 * np.sin(x / 30)),
                int(150 + 50 * np.cos(y / 20)),
                int(200 - 80 * np.sin((x + y) / 40)),
            )

    prompt = "Describe what you see in this image in detail."

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt},
    ]}]

    formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Use process_vision_info for this PoC (still installed)
    from mlx_vlm.video_generate import process_vision_info
    image_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)

    inputs = processor(text=[formatted], images=image_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in inputs.items()}

    input_ids = mx.array(np.asarray(inputs["input_ids"]))
    pixel_values = mx.array(np.asarray(inputs.get("pixel_values", np.array([]))))
    mask_arr = mx.array(np.asarray(inputs["attention_mask"]))

    kwargs = {}
    if "image_grid_thw" in inputs:
        kwargs["image_grid_thw"] = mx.array(np.asarray(inputs["image_grid_thw"]))

    # Get embeddings
    embedding_output = model.get_input_embeddings(input_ids, pixel_values, mask=mask_arr, **kwargs)
    inputs_embeds = embedding_output.inputs_embeds
    kwargs.update({
        k: v for k, v in embedding_output.to_dict().items()
        if k != "inputs_embeds" and v is not None
    })

    # Identify visual vs text positions
    ids_np = np.array(input_ids[0])
    image_token_id = model.config.image_token_id
    video_token_id = model.config.video_token_id
    visual_mask = (ids_np == image_token_id) | (ids_np == video_token_id)
    text_mask = ~visual_mask

    n_total = len(ids_np)
    n_visual = visual_mask.sum()
    n_text = text_mask.sum()
    vis_idx = np.where(visual_mask)[0]
    txt_idx = np.where(text_mask)[0]

    print(f"Tokens: {n_total} total, {n_visual} visual, {n_text} text")

    # Method: Run hidden states through each layer's Q/K projections
    # to compute attention scores WITHOUT modifying the model
    lm = model.language_model
    layers = lm.model.layers
    n_layers = len(layers)

    # Run full prefill to get hidden states at each layer
    # We'll intercept by running layer-by-layer manually
    print(f"\nRunning layer-by-layer analysis ({n_layers} layers)...")

    # Get the input to the transformer layers
    h = inputs_embeds  # (1, L, D)
    # Apply input layernorm if present (Qwen uses RMSNorm before layers? No, usually after embed)

    # We need position_ids for RoPE
    if "image_grid_thw" in kwargs:
        new_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        position_ids, _ = lm.get_rope_index(input_ids, attention_mask=new_mask, **kwargs)
    else:
        position_ids = None

    # Create attention mask
    from mlx_vlm.models.base import create_attention_mask
    from trio_core.generate import make_prompt_cache

    prompt_cache = make_prompt_cache(lm)
    attn_mask = create_attention_mask(h, prompt_cache)

    target_layers = [0, 1, 2, 4, 8, 16, 24, 32, 35]
    target_layers = [i for i in target_layers if i < n_layers]

    results = []

    for i in range(n_layers):
        layer = layers[i]
        attn = layer.self_attn

        if i in target_layers:
            # Compute Q, K manually
            residual = h
            h_norm = layer.input_layernorm(h)

            B, L, D = h_norm.shape
            queries = attn.q_proj(h_norm).reshape(B, L, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
            keys = attn.k_proj(h_norm).reshape(B, L, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)

            # Apply RoPE
            from mlx_vlm.models.qwen2_5_vl.language import apply_multimodal_rotary_pos_emb
            values_dummy = keys  # just for shape
            cos, sin = attn.rotary_emb(values_dummy, position_ids)
            queries_rot, keys_rot = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin, unqueeze_dim=1)

            # GQA: repeat K to match Q heads
            n_heads = queries_rot.shape[1]
            n_kv = keys_rot.shape[1]
            if n_kv < n_heads:
                keys_rot = mx.repeat(keys_rot, n_heads // n_kv, axis=1)

            # Compute attention scores
            scores = (queries_rot @ keys_rot.transpose(0, 1, 3, 2)) * attn.scale
            if attn_mask is not None:
                scores = scores + attn_mask
            weights = mx.softmax(scores, axis=-1)
            mx.eval(weights)

            analyze_distribution(np.array(weights[0]), vis_idx, txt_idx, i, n_visual)
            results.append(i)

        # Actually run the layer to get correct hidden state for next layer
        out = layer(h, mask=attn_mask, cache=prompt_cache[i], position_ids=position_ids)
        # Qwen layers return just hidden_states
        h = out
        mx.eval(h)
        # Update mask for next layer (cache grew)
        attn_mask = create_attention_mask(h, prompt_cache)

    # Also try a second prompt to check if distribution changes
    print("\n" + "=" * 60)
    print("SECOND PROMPT: 'Is there any text in this image?'")
    print("=" * 60)

    prompt2 = "Is there any text in this image?"
    messages2 = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt2},
    ]}]
    formatted2 = processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
    image_inputs2, _, _ = process_vision_info(messages2, return_video_kwargs=True)
    inputs2 = processor(text=[formatted2], images=image_inputs2, padding=True, return_tensors="pt")
    inputs2 = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in inputs2.items()}

    input_ids2 = mx.array(np.asarray(inputs2["input_ids"]))
    pixel_values2 = mx.array(np.asarray(inputs2.get("pixel_values", np.array([]))))
    mask_arr2 = mx.array(np.asarray(inputs2["attention_mask"]))

    kwargs2 = {}
    if "image_grid_thw" in inputs2:
        kwargs2["image_grid_thw"] = mx.array(np.asarray(inputs2["image_grid_thw"]))

    embedding_output2 = model.get_input_embeddings(input_ids2, pixel_values2, mask=mask_arr2, **kwargs2)
    inputs_embeds2 = embedding_output2.inputs_embeds
    kwargs2.update({
        k: v for k, v in embedding_output2.to_dict().items()
        if k != "inputs_embeds" and v is not None
    })

    ids_np2 = np.array(input_ids2[0])
    visual_mask2 = (ids_np2 == image_token_id) | (ids_np2 == video_token_id)
    vis_idx2 = np.where(visual_mask2)[0]
    txt_idx2 = np.where(~visual_mask2)[0]
    n_visual2 = visual_mask2.sum()

    if "image_grid_thw" in kwargs2:
        new_mask2 = mx.ones(input_ids2.shape, dtype=mx.int32)
        position_ids2, _ = lm.get_rope_index(input_ids2, attention_mask=new_mask2, **kwargs2)
    else:
        position_ids2 = None

    prompt_cache2 = make_prompt_cache(lm)
    h2 = inputs_embeds2
    attn_mask2 = create_attention_mask(h2, prompt_cache2)

    for i in range(n_layers):
        layer = layers[i]
        attn = layer.self_attn

        if i in [2]:  # Just check layer 2 for comparison
            residual = h2
            h_norm = layer.input_layernorm(h2)
            B, L, D = h_norm.shape
            queries = attn.q_proj(h_norm).reshape(B, L, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
            keys = attn.k_proj(h_norm).reshape(B, L, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
            values_dummy = keys
            cos, sin = attn.rotary_emb(values_dummy, position_ids2)
            queries_rot, keys_rot = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin, unqueeze_dim=1)
            n_heads = queries_rot.shape[1]
            n_kv = keys_rot.shape[1]
            if n_kv < n_heads:
                keys_rot = mx.repeat(keys_rot, n_heads // n_kv, axis=1)
            scores = (queries_rot @ keys_rot.transpose(0, 1, 3, 2)) * attn.scale
            if attn_mask2 is not None:
                scores = scores + attn_mask2
            weights = mx.softmax(scores, axis=-1)
            mx.eval(weights)
            analyze_distribution(np.array(weights[0]), vis_idx2, txt_idx2, i, n_visual2)

        out = layer(h2, mask=attn_mask2, cache=prompt_cache2[i], position_ids=position_ids2)
        h2 = out
        mx.eval(h2)
        attn_mask2 = create_attention_mask(h2, prompt_cache2)


def analyze_distribution(w, vis_idx, txt_idx, layer_idx, n_visual):
    """Analyze text→visual attention distribution.

    w: (n_heads, L, L) attention weights
    """
    if len(vis_idx) == 0 or len(txt_idx) == 0:
        print(f"  Layer {layer_idx}: No visual or text tokens found")
        return

    # text→visual attention: for each text query, attention to visual keys
    t2v = w[:, txt_idx][:, :, vis_idx]  # (n_heads, n_text, n_visual)

    # Average across text tokens and heads
    avg_per_visual = t2v.mean(axis=(0, 1))  # (n_visual,)
    total = avg_per_visual.sum()
    if total < 1e-10:
        print(f"  Layer {layer_idx}: Near-zero text→visual attention")
        return

    avg_per_visual = avg_per_visual / total

    # Sort descending
    sorted_attn = np.sort(avg_per_visual)[::-1]
    cumsum = np.cumsum(sorted_attn)

    # Metrics
    n_v = len(vis_idx)
    entropy = -np.sum(avg_per_visual * np.log(avg_per_visual + 1e-10)) / np.log(n_v)
    top10_pct = cumsum[max(0, n_v // 10 - 1)]
    top25_pct = cumsum[max(0, n_v // 4 - 1)]
    top50_pct = cumsum[max(0, n_v // 2 - 1)]

    # Gini coefficient
    sorted_asc = np.sort(avg_per_visual)
    index = np.arange(1, n_v + 1)
    gini = (2 * np.sum(index * sorted_asc) / (n_v * np.sum(sorted_asc))) - (n_v + 1) / n_v

    # Tokens needed for 80%/90%
    pct80 = np.searchsorted(cumsum, 0.8) + 1
    pct90 = np.searchsorted(cumsum, 0.9) + 1

    # Total text→visual vs text→text attention ratio
    t2t = w[:, txt_idx][:, :, txt_idx].mean()
    t2v_total = w[:, txt_idx][:, :, vis_idx].mean()

    print(f"  Layer {layer_idx}: {n_v} visual tokens")
    print(f"    Text→Visual ratio:    {t2v_total/(t2v_total+t2t)*100:.1f}% of text attention goes to visual")
    print(f"    Entropy (normalized): {entropy:.3f}  (1.0=uniform, 0.0=concentrated)")
    print(f"    Gini coefficient:     {gini:.3f}  (0.0=equal, 1.0=one token gets all)")
    print(f"    Top 10% tokens hold:  {top10_pct*100:.1f}% of attention")
    print(f"    Top 25% tokens hold:  {top25_pct*100:.1f}% of attention")
    print(f"    Top 50% tokens hold:  {top50_pct*100:.1f}% of attention")
    print(f"    Tokens for 80% attn:  {pct80}/{n_v} ({pct80/n_v*100:.0f}%)")
    print(f"    Tokens for 90% attn:  {pct90}/{n_v} ({pct90/n_v*100:.0f}%)")
    print()


if __name__ == "__main__":
    main()

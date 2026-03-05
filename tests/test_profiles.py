"""Tests for trio_core.profiles — model-specific architecture parameters."""

from trio_core.profiles import ModelProfile, get_profile, PROFILES


class TestModelProfile:
    def test_merge_factor_qwen25vl(self):
        p = PROFILES["qwen2.5-vl-3b"]
        assert p.merge_factor == 28  # 14 × 2

    def test_merge_factor_qwen35(self):
        p = PROFILES["qwen3.5-0.8b"]
        assert p.merge_factor == 32  # 16 × 2

    def test_compute_visual_tokens(self):
        p = PROFILES["qwen2.5-vl-3b"]
        # 8 frames, 224×224 → (8/2) × (224/28) × (224/28) = 4 × 8 × 8 = 256
        tokens = p.compute_visual_tokens(8, 224, 224)
        assert tokens == 256

    def test_compute_visual_tokens_qwen35(self):
        p = PROFILES["qwen3.5-0.8b"]
        # 4 frames, 320×320 → (4/2) × (320/32) × (320/32) = 2 × 10 × 10 = 200
        tokens = p.compute_visual_tokens(4, 320, 320)
        assert tokens == 200

    def test_compute_optimal_params_within_budget(self):
        p = PROFILES["qwen2.5-vl-3b"]
        frames, h, w = p.compute_optimal_params(5.0, 224, 224)
        tokens = p.compute_visual_tokens(frames, h, w)
        assert tokens <= p.max_visual_tokens
        assert frames % p.temporal_patch == 0
        assert h % p.merge_factor == 0
        assert w % p.merge_factor == 0

    def test_compute_optimal_params_high_res_scales_down(self):
        p = PROFILES["qwen2.5-vl-3b"]
        frames, h, w = p.compute_optimal_params(30.0, 1080, 1920)
        tokens = p.compute_visual_tokens(frames, h, w)
        assert tokens <= p.max_visual_tokens
        # Should have scaled down from native
        assert h < 1080 or w < 1920


class TestGetProfile:
    def test_direct_lookup(self):
        p = get_profile("qwen2.5-vl-3b")
        assert p.family == "qwen2.5-vl"

    def test_huggingface_id(self):
        p = get_profile("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
        assert p.family == "qwen2.5-vl"
        assert p.param_size == "3B"

    def test_qwen35_pattern(self):
        p = get_profile("Qwen/Qwen3.5-0.8B")
        assert p.family == "qwen3.5"

    def test_qwen25_7b_pattern(self):
        p = get_profile("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
        assert p.param_size == "7B"

    def test_unknown_fallback(self):
        p = get_profile("some-unknown-model")
        assert p.family == "qwen2.5-vl"  # fallback to 3B
        assert p.param_size == "3B"

    def test_case_insensitive(self):
        p = get_profile("QWEN2.5-VL-3B")
        assert p.family == "qwen2.5-vl"

"""Test counting logic — debounce + cumulative tracking."""

import pytest

cv2 = pytest.importorskip("cv2")
from trio_core._webcam_gui import _parse_counts


class TestParseCounts:
    def test_basic(self):
        assert _parse_counts("COUNT people:2 cars:1 dogs:0 cats:0") == {
            "people": 2,
            "cars": 1,
            "dogs": 0,
            "cats": 0,
        }

    def test_case_insensitive(self):
        assert _parse_counts("COUNT People:3 Cars:0 Dogs:1 Cats:0") == {
            "people": 3,
            "cars": 0,
            "dogs": 1,
            "cats": 0,
        }

    def test_missing_keys(self):
        assert _parse_counts("I see a person walking") == {
            "people": 0,
            "cars": 0,
            "dogs": 0,
            "cats": 0,
        }

    def test_embedded_in_text(self):
        assert _parse_counts("A man walks his dog. COUNT people:1 cars:2 dogs:1 cats:0") == {
            "people": 1,
            "cars": 2,
            "dogs": 1,
            "cats": 0,
        }


KEYS = ["people", "cars", "dogs", "cats"]
DEBOUNCE_FRAMES = 3


def _simulate_counting(frame_counts: list[dict]) -> dict:
    """Simulate the cumulative counting logic from _webcam_gui.

    Args:
        frame_counts: list of per-frame visible counts, e.g.
            [{"people": 1}, {"people": 1}, {"people": 0}, {"people": 1}]

    Returns:
        Final cumulative total_counts.
    """
    total_counts = {k: 0 for k in KEYS}
    prev_visible = {k: 0 for k in KEYS}
    zero_streak = {k: 0 for k in KEYS}

    for fc in frame_counts:
        current = {k: fc.get(k, 0) for k in KEYS}
        for key in KEYS:
            if current[key] > 0:
                zero_streak[key] = 0
                if current[key] > prev_visible[key]:
                    total_counts[key] += current[key] - prev_visible[key]
                    prev_visible[key] = current[key]
                # On decrease: keep prev_visible as high-water mark
            else:
                zero_streak[key] += 1
                if zero_streak[key] >= DEBOUNCE_FRAMES:
                    prev_visible[key] = 0

    return total_counts


class TestCumulativeCounting:
    def test_single_person_stays(self):
        """Person visible for 5 frames — should count as 1."""
        frames = [{"people": 1}] * 5
        assert _simulate_counting(frames)["people"] == 1

    def test_person_with_flicker(self):
        """Person visible, model flickers to 0 for 1 frame, then back.
        Debounce should prevent double-count."""
        frames = [
            {"people": 1},
            {"people": 1},
            {"people": 0},  # flicker
            {"people": 1},
            {"people": 1},
        ]
        assert _simulate_counting(frames)["people"] == 1

    def test_person_leaves_then_new_person(self):
        """Person A leaves (3+ zero frames), person B arrives. Should count 2."""
        frames = (
            [{"people": 1}] * 3  # person A
            + [{"people": 0}] * 4  # gone (exceeds debounce)
            + [{"people": 1}] * 3  # person B
        )
        assert _simulate_counting(frames)["people"] == 2

    def test_two_people_arrive_sequentially(self):
        """1 person, then 2 people. Should count 2 total."""
        frames = [
            {"people": 1},
            {"people": 1},
            {"people": 2},
            {"people": 2},
        ]
        assert _simulate_counting(frames)["people"] == 2

    def test_person_leaves_no_debounce_complete(self):
        """Person leaves but only 2 zero frames (< debounce). New person arrives.
        Should NOT double-count."""
        frames = (
            [{"people": 1}] * 3
            + [{"people": 0}] * 2  # less than debounce threshold
            + [{"people": 1}] * 3
        )
        assert _simulate_counting(frames)["people"] == 1

    def test_multiple_categories(self):
        """People and dogs counted independently."""
        frames = [
            {"people": 1, "dogs": 1},
            {"people": 1, "dogs": 1},
            {"people": 2, "dogs": 0},  # dog flickers
            {"people": 2, "dogs": 1},
        ]
        result = _simulate_counting(frames)
        assert result["people"] == 2
        assert result["dogs"] == 1

    def test_empty_stream(self):
        """No objects ever seen."""
        frames = [{"people": 0}] * 5
        assert _simulate_counting(frames)["people"] == 0

    def test_crowd_fluctuation(self):
        """Crowd: 3 → 2 → 4 → 3. Peak was 4, total should be 4."""
        frames = [
            {"people": 3},
            {"people": 2},
            {"people": 4},
            {"people": 3},
        ]
        # 0→3 = +3, 3→2 no add, 2→4 = +2, total = 5? No...
        # Actually: prev starts 0, frame1: 3>0 → total+=3, prev=3
        # frame2: 2>0 but 2<3 → no add, prev=3
        # frame3: 4>3 → total+=1, prev=4
        # frame4: 3<4 → no add, prev=4
        # total = 4
        assert _simulate_counting(frames)["people"] == 4

    def test_starbucks_scenario(self):
        """Realistic: steady stream of customers entering/exiting Starbucks.
        Each 'frame' is a 3s snapshot. Some overlap, some gaps.

        Ground truth: 5 distinct people enter over ~60s.
        """
        frames = [
            {"people": 0},  # empty
            {"people": 1},  # customer A enters
            {"people": 1},  # A at counter
            {"people": 2},  # B enters while A still there
            {"people": 2},  # both inside
            {"people": 1},  # A leaves
            {"people": 1},  # B alone
            {"people": 0},  # B leaves (flicker - just 1 frame)
            {"people": 1},  # B still there actually (debounce saves us)
            {"people": 0},  # now B really leaving
            {"people": 0},  # ...
            {"people": 0},  # 3 zeros → debounce clears
            {"people": 1},  # C enters
            {"people": 2},  # D enters
            {"people": 3},  # E enters
            {"people": 2},  # C leaves
            {"people": 1},  # D leaves
            {"people": 0},  # E leaves
        ]
        result = _simulate_counting(frames)
        # A+B = 2, then after debounce reset, C→D→E adds peak of 3 = total 5
        assert result["people"] == 5

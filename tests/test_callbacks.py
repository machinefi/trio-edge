"""Tests for trio_core.callbacks."""

from trio_core.callbacks import EVENTS, CallbackMixin, get_default_callbacks


class DummyEngine(CallbackMixin):
    def __init__(self):
        self._init_callbacks()
        self.value = 0


class TestCallbackMixin:
    def test_default_callbacks_has_all_events(self):
        cbs = get_default_callbacks()
        for event in EVENTS:
            assert event in cbs

    def test_add_and_run_callback(self):
        engine = DummyEngine()
        engine.add_callback("on_vlm_end", lambda e: setattr(e, "value", 42))
        engine.run_callbacks("on_vlm_end")
        assert engine.value == 42

    def test_multiple_callbacks(self):
        engine = DummyEngine()
        engine.add_callback("on_vlm_start", lambda e: setattr(e, "value", e.value + 1))
        engine.add_callback("on_vlm_start", lambda e: setattr(e, "value", e.value + 10))
        engine.run_callbacks("on_vlm_start")
        assert engine.value == 11

    def test_callback_error_does_not_propagate(self):
        engine = DummyEngine()
        engine.add_callback("on_vlm_end", lambda e: 1 / 0)  # ZeroDivisionError
        engine.add_callback("on_vlm_end", lambda e: setattr(e, "value", 99))
        engine.run_callbacks("on_vlm_end")  # should not raise
        assert engine.value == 99

    def test_clear_specific_event(self):
        engine = DummyEngine()
        engine.add_callback("on_vlm_end", lambda e: setattr(e, "value", 1))
        engine.clear_callbacks("on_vlm_end")
        engine.run_callbacks("on_vlm_end")
        assert engine.value == 0

    def test_clear_all(self):
        engine = DummyEngine()
        engine.add_callback("on_vlm_end", lambda e: setattr(e, "value", 1))
        engine.add_callback("on_vlm_start", lambda e: setattr(e, "value", 2))
        engine.clear_callbacks()
        engine.run_callbacks("on_vlm_end")
        engine.run_callbacks("on_vlm_start")
        assert engine.value == 0

    def test_unknown_event_warns(self, caplog):
        import logging

        engine = DummyEngine()
        with caplog.at_level(logging.WARNING):
            engine.add_callback("on_fake_event", lambda e: None)
        assert "Unknown callback event" in caplog.text

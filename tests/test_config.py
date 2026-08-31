import importlib

import pytest

import config


class TestValidate:
    def _set_valid_defaults(self, monkeypatch):
        monkeypatch.setattr(config, "HF_TOKEN", "fake-token")
        monkeypatch.setattr(config, "TOP_K", 3)
        monkeypatch.setattr(config, "TEMPERATURE", 0.7)
        monkeypatch.setattr(config, "SCORE_THRESHOLD", 0.3)
        monkeypatch.setattr(config, "MAX_NEW_TOKENS", 512)
        monkeypatch.setattr(config, "LLM_MAX_RETRIES", 2)
        monkeypatch.setattr(config, "MAX_HISTORY_TURNS", 10)

    def test_passes_with_valid_defaults(self, monkeypatch):
        self._set_valid_defaults(monkeypatch)
        config.validate()  # should not raise

    def test_raises_when_token_missing(self, monkeypatch):
        self._set_valid_defaults(monkeypatch)
        monkeypatch.setattr(config, "HF_TOKEN", "")
        with pytest.raises(config.ConfigError, match="HF_TOKEN"):
            config.validate()

    def test_raises_when_top_k_not_positive(self, monkeypatch):
        self._set_valid_defaults(monkeypatch)
        monkeypatch.setattr(config, "TOP_K", 0)
        with pytest.raises(config.ConfigError, match="TOP_K"):
            config.validate()

    def test_raises_when_temperature_out_of_range(self, monkeypatch):
        self._set_valid_defaults(monkeypatch)
        monkeypatch.setattr(config, "TEMPERATURE", 5.0)
        with pytest.raises(config.ConfigError, match="TEMPERATURE"):
            config.validate()

    def test_raises_when_score_threshold_out_of_range(self, monkeypatch):
        self._set_valid_defaults(monkeypatch)
        monkeypatch.setattr(config, "SCORE_THRESHOLD", -0.1)
        with pytest.raises(config.ConfigError, match="SCORE_THRESHOLD"):
            config.validate()

    def test_collects_multiple_errors(self, monkeypatch):
        self._set_valid_defaults(monkeypatch)
        monkeypatch.setattr(config, "HF_TOKEN", "")
        monkeypatch.setattr(config, "TOP_K", -1)
        with pytest.raises(config.ConfigError) as exc_info:
            config.validate()
        message = str(exc_info.value)
        assert "HF_TOKEN" in message
        assert "TOP_K" in message

    def test_import_does_not_raise_without_token(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "")
        importlib.reload(config)  # should not raise
        try:
            assert config.HF_TOKEN == ""
        finally:
            # Undo the env patch now (not just at test teardown) and reload
            # again so later tests see config matching the real environment.
            monkeypatch.undo()
            importlib.reload(config)

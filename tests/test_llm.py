from unittest.mock import MagicMock, patch

import config
import llm


SAMPLE_RETRIEVED = [
    {
        "name": "Pasta Carbonara",
        "cuisine": "Italian",
        "ingredients": "pasta, eggs, bacon, cheese",
        "directions": "boil pasta, mix eggs",
        "url": "http://example.com/carbonara",
        "score": 0.95,
    }
]


class TestBuildMessages:
    def test_starts_with_system(self):
        msgs = llm.build_messages("hi", [], [])
        assert msgs[0]["role"] == "system"

    def test_no_retrieved_no_context_message(self):
        msgs = llm.build_messages("hi", [], [])
        roles = [m["role"] for m in msgs]
        assert roles.count("system") == 1

    def test_retrieved_adds_second_system_message(self):
        msgs = llm.build_messages("make pasta", SAMPLE_RETRIEVED, [])
        system_msgs = [m for m in msgs if m["role"] == "system"]
        assert len(system_msgs) == 2
        assert "Pasta Carbonara" in system_msgs[1]["content"]

    def test_user_message_is_last(self):
        msgs = llm.build_messages("what can I cook", SAMPLE_RETRIEVED, [])
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "what can I cook"

    def test_history_inserted_before_user(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        msgs = llm.build_messages("a follow-up", [], history)
        roles = [m["role"] for m in msgs]
        # system, user(history), assistant(history), user(current)
        assert roles == ["system", "user", "assistant", "user"]

    def test_order_system_then_retrieved_then_history_then_user(self):
        history = [{"role": "user", "content": "prev"}, {"role": "assistant", "content": "resp"}]
        msgs = llm.build_messages("now", SAMPLE_RETRIEVED, history)
        roles = [m["role"] for m in msgs]
        assert roles[0] == "system"
        assert roles[1] == "system"   # retrieved context
        assert roles[-1] == "user"

    def test_retrieved_context_includes_directions(self):
        msgs = llm.build_messages("pasta", SAMPLE_RETRIEVED, [])
        context_msg = [m for m in msgs if m["role"] == "system" and "Retrieved" in m["content"]][0]
        assert "boil pasta" in context_msg["content"]

    def test_retrieved_context_includes_url(self):
        msgs = llm.build_messages("pasta", SAMPLE_RETRIEVED, [])
        context_msg = [m for m in msgs if m["role"] == "system" and "Retrieved" in m["content"]][0]
        assert "http://example.com/carbonara" in context_msg["content"]

    def test_history_truncated_to_max_turns(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_HISTORY_TURNS", 3)
        history = []
        for i in range(15):
            history.append({"role": "user", "content": f"user turn {i}"})
            history.append({"role": "assistant", "content": f"assistant turn {i}"})

        msgs = llm.build_messages("now", [], history)
        history_msgs = msgs[1:-1]  # drop leading system + trailing current user

        assert len(history_msgs) == 6  # 3 turns * 2 messages
        assert history_msgs == history[-6:]

    def test_history_under_limit_not_truncated(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_HISTORY_TURNS", 10)
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        msgs = llm.build_messages("now", [], history)
        assert msgs[1:-1] == history

    def test_zero_max_history_turns_drops_all_history(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_HISTORY_TURNS", 0)
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        msgs = llm.build_messages("now", [], history)
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user"]


class TestStreamResponse:
    def _make_chunk(self, token: str):
        chunk = MagicMock()
        chunk.choices[0].delta.content = token
        return chunk

    def _make_empty_choices_chunk(self):
        chunk = MagicMock()
        chunk.choices = []
        return chunk

    def test_yields_tokens(self):
        chunks = [self._make_chunk(t) for t in ["Hello", " world", "!"]]
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = iter(chunks)

        with patch.object(llm, "_get_client", return_value=mock_client):
            tokens = list(llm.stream_response([{"role": "user", "content": "hi"}]))

        assert tokens == ["Hello", " world", "!"]

    def test_skips_none_tokens(self):
        chunks = [self._make_chunk("hi"), self._make_chunk(None), self._make_chunk("!")]
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = iter(chunks)

        with patch.object(llm, "_get_client", return_value=mock_client):
            tokens = list(llm.stream_response([{"role": "user", "content": "hi"}]))

        assert tokens == ["hi", "!"]

    def test_skips_chunk_with_empty_choices(self):
        # Some providers (e.g. Featherless AI) send a trailing chunk with an
        # empty choices list (usage stats) to mark end-of-stream.
        chunks = [self._make_chunk("hi"), self._make_empty_choices_chunk(), self._make_chunk("!")]
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = iter(chunks)

        with patch.object(llm, "_get_client", return_value=mock_client):
            tokens = list(llm.stream_response([{"role": "user", "content": "hi"}]))

        assert tokens == ["hi", "!"]

    def test_rate_limit_yields_friendly_message(self):
        from huggingface_hub.errors import HfHubHTTPError
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        error = HfHubHTTPError("rate limit", response=mock_resp)

        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = error

        with patch.object(llm, "_get_client", return_value=mock_client), patch("llm.time.sleep"):
            tokens = list(llm.stream_response([]))

        assert len(tokens) == 1
        assert "rate-limited" in tokens[0].lower()

    def test_model_loading_yields_friendly_message(self):
        from huggingface_hub.errors import HfHubHTTPError
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        error = HfHubHTTPError("loading", response=mock_resp)

        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = error

        with patch.object(llm, "_get_client", return_value=mock_client), patch("llm.time.sleep"):
            tokens = list(llm.stream_response([]))

        assert len(tokens) == 1
        assert "loading" in tokens[0].lower()

    def test_retries_after_transient_429_then_succeeds(self, monkeypatch):
        from huggingface_hub.errors import HfHubHTTPError
        monkeypatch.setattr(config, "LLM_MAX_RETRIES", 2)
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        error = HfHubHTTPError("rate limit", response=mock_resp)

        chunks = [self._make_chunk(t) for t in ["Hello", " world"]]
        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = [error, iter(chunks)]

        with patch.object(llm, "_get_client", return_value=mock_client), patch("llm.time.sleep") as mock_sleep:
            tokens = list(llm.stream_response([{"role": "user", "content": "hi"}]))

        assert tokens == ["Hello", " world"]
        assert mock_client.chat_completion.call_count == 2
        mock_sleep.assert_called_once()

    def test_exhausts_retries_still_yields_friendly_message(self, monkeypatch):
        from huggingface_hub.errors import HfHubHTTPError
        monkeypatch.setattr(config, "LLM_MAX_RETRIES", 2)
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        error = HfHubHTTPError("rate limit", response=mock_resp)

        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = [error, error, error]

        with patch.object(llm, "_get_client", return_value=mock_client), patch("llm.time.sleep"):
            tokens = list(llm.stream_response([]))

        assert len(tokens) == 1
        assert "rate-limited" in tokens[0].lower()
        assert mock_client.chat_completion.call_count == 3  # initial + 2 retries

    def test_no_retry_after_partial_tokens_yielded(self, monkeypatch):
        from huggingface_hub.errors import HfHubHTTPError
        monkeypatch.setattr(config, "LLM_MAX_RETRIES", 2)
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        error = HfHubHTTPError("loading", response=mock_resp)

        def flaky_stream():
            yield self._make_chunk("Hello")
            raise error

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = flaky_stream()

        with patch.object(llm, "_get_client", return_value=mock_client), patch("llm.time.sleep"):
            tokens = list(llm.stream_response([]))

        assert tokens[0] == "Hello"
        assert len(tokens) == 2
        assert "loading" in tokens[1].lower()
        assert mock_client.chat_completion.call_count == 1

    def test_backoff_increases_between_retries(self, monkeypatch):
        from huggingface_hub.errors import HfHubHTTPError
        monkeypatch.setattr(config, "LLM_MAX_RETRIES", 2)
        monkeypatch.setattr(config, "LLM_RETRY_BACKOFF_BASE", 1.0)
        monkeypatch.setattr(config, "LLM_RETRY_BACKOFF_MAX", 8.0)
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        error = HfHubHTTPError("rate limit", response=mock_resp)

        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = [error, error, error]

        with patch.object(llm, "_get_client", return_value=mock_client), patch("llm.time.sleep") as mock_sleep:
            list(llm.stream_response([]))

        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0]

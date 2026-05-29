from unittest.mock import MagicMock, patch

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

    def test_retrieved_context_includes_url(self):
        msgs = llm.build_messages("pasta", SAMPLE_RETRIEVED, [])
        context_msg = [m for m in msgs if m["role"] == "system" and "Retrieved" in m["content"]][0]
        assert "http://example.com/carbonara" in context_msg["content"]


class TestStreamResponse:
    def _make_chunk(self, token: str):
        chunk = MagicMock()
        chunk.choices[0].delta.content = token
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

    def test_rate_limit_yields_friendly_message(self):
        from huggingface_hub.errors import HfHubHTTPError
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        error = HfHubHTTPError("rate limit", response=mock_resp)

        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = error

        with patch.object(llm, "_get_client", return_value=mock_client):
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

        with patch.object(llm, "_get_client", return_value=mock_client):
            tokens = list(llm.stream_response([]))

        assert len(tokens) == 1
        assert "loading" in tokens[0].lower()

import eval as eval_module


SAMPLE_RECIPE = {
    "name": "Pasta Carbonara",
    "ingredients": "pasta, eggs, bacon, cheese",
    "directions": "boil pasta, mix eggs",
    "url": "http://example.com/carbonara",
    "cuisine": "Italian",
    "score": 0.95,
}


class TestFormatForJudge:
    def test_includes_directions_in_judge_context(self):
        text = eval_module._format_for_judge(SAMPLE_RECIPE)
        assert "boil pasta, mix eggs" in text
        assert "Pasta Carbonara" in text

    def test_omits_directions_line_when_blank(self):
        recipe = dict(SAMPLE_RECIPE, directions="")
        text = eval_module._format_for_judge(recipe)
        assert "Directions" not in text

    def test_omits_directions_line_when_missing(self):
        recipe = {k: v for k, v in SAMPLE_RECIPE.items() if k != "directions"}
        text = eval_module._format_for_judge(recipe)
        assert "Directions" not in text

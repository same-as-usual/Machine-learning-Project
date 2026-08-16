from manipulens.models.neutralize import MASK_TOKEN, mask_entities


def test_single_entities_masked():
    assert mask_entities("Biden announces plan") == f"{MASK_TOKEN} announces plan"
    assert mask_entities("Trump announces plan") == f"{MASK_TOKEN} announces plan"


def test_swap_produces_identical_text():
    template = "{e} responds to questions about the budget"
    for a, b in [("Biden", "Trump"), ("Democrats", "Republicans"), ("liberal", "conservative")]:
        assert mask_entities(template.format(e=a)) == mask_entities(template.format(e=b))


def test_multiword_phrases_masked():
    assert MASK_TOKEN in mask_entities("The Left slams the budget")
    assert mask_entities("the left slams X") == mask_entities("the right slams X")


def test_case_insensitive():
    assert mask_entities("TRUMP speaks") == f"{MASK_TOKEN} speaks"


def test_word_boundaries_respected():
    # 'left' alone is not in the lexicon as bare directional usage is common,
    # but ensure no substring mangling of ordinary words
    assert mask_entities("The team left the stadium") == "The team left the stadium"
    assert mask_entities("A conservatively invested fund") == "A conservatively invested fund"


def test_nonpolitical_text_unchanged():
    text = "Scientists report progress on malaria vaccine trial"
    assert mask_entities(text) == text


def test_masking_idempotent():
    once = mask_entities("Biden slams Trump")
    assert mask_entities(once) == once

from manipulens.data.dedup import assign_dup_groups, char_shingles, normalize


def test_normalize_strips_punct_and_case():
    assert normalize("You WON'T Believe!!") == "you wont believe"


def test_char_shingles_short_text():
    assert char_shingles("ab", 3) == {b"ab"}


def test_near_duplicates_share_group():
    headlines = [
        "Senate passes major infrastructure bill in landmark 69-30 vote",
        "Senate passes major infrastructure bill in landmark 69–30 vote!",  # punctuation variant
        "Ten ways to improve your sleep quality tonight",
    ]
    groups = assign_dup_groups(headlines, num_perm=128, threshold=0.8, k=3)
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]


def test_distinct_headlines_distinct_groups():
    headlines = [
        "Federal Reserve holds interest rates steady",
        "Local library reopens after major renovation",
        "Storm system expected to bring rain to the Midwest",
    ]
    groups = assign_dup_groups(headlines, num_perm=128, threshold=0.8, k=3)
    assert len(set(groups)) == 3

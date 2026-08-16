import pandas as pd

try:
    import pandera.pandas as pa
except ImportError:
    import pandera as pa
import pytest

pytestmark = pytest.mark.data

from manipulens.data.schemas import validate_raw, validate_split  # noqa: E402


def _raw_df():
    return pd.DataFrame(
        {
            "headline": ["Some headline", "Another headline"],
            "label_clickbait": [0, 1],
            "source": ["test", "test"],
            "outlet": ["", ""],
        }
    )


def test_valid_raw_passes():
    validate_raw(_raw_df())


def test_bad_label_fails():
    df = _raw_df()
    df.loc[0, "label_clickbait"] = 5
    with pytest.raises(pa.errors.SchemaError):
        validate_raw(df)


def test_empty_headline_fails():
    df = _raw_df()
    df.loc[0, "headline"] = ""
    with pytest.raises(pa.errors.SchemaError):
        validate_raw(df)


def test_split_schema_rejects_unknown_split():
    df = _raw_df().assign(dup_group=[0, 1], split=["train", "holdout"])
    with pytest.raises(pa.errors.SchemaError):
        validate_split(df)

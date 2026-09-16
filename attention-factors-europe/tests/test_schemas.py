"""The contract in docs/schemas.md is enforced, and violations are actually caught."""

import pandas as pd
import pytest

from afe import schemas


def test_fixture_tables_honour_the_contract(returns, universe, features):
    schemas.validate(returns, schemas.RETURNS)
    schemas.validate(universe, schemas.UNIVERSE)
    schemas.validate(features, schemas.FEATURES)


def test_feature_count_is_two_per_characteristic_plus_rf(features):
    chars = [c for c in features.columns if c.startswith("char_")]
    meds = [c for c in features.columns if c.startswith("med_")]
    assert len(chars) == len(meds)
    assert len(schemas.feature_columns(features)) == 2 * len(chars) + 1


def test_duplicate_keys_are_rejected(returns):
    broken = pd.concat([returns, returns.head(1)], ignore_index=True)
    broken = broken.sort_values(["date", "sec_id"], ignore_index=True)
    with pytest.raises(schemas.SchemaError, match="duplicate"):
        schemas.validate(broken, schemas.RETURNS)


def test_unsorted_rows_are_rejected(returns):
    with pytest.raises(schemas.SchemaError, match="not sorted"):
        schemas.validate(returns.iloc[::-1].reset_index(drop=True), schemas.RETURNS)


def test_wrong_dtype_is_rejected(returns):
    broken = returns.copy()
    broken["ret"] = broken["ret"].astype("float64")
    with pytest.raises(schemas.SchemaError, match="dtype"):
        schemas.validate(broken, schemas.RETURNS)


def test_nulls_in_required_columns_are_rejected(returns):
    broken = returns.copy()
    broken.loc[0, "ret"] = float("nan")
    with pytest.raises(schemas.SchemaError, match="nulls"):
        schemas.validate(broken, schemas.RETURNS)

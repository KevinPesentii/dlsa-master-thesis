"""Universe construction is the first-order risk. Long & Xiao (2024) was withdrawn for it."""

import pandas as pd
import pytest

from afe import schemas
from afe.data import synthetic

from conftest import UNIVERSE_SIZE


def test_exactly_n_names_every_month(universe):
    schemas.validate_universe(universe, size=UNIVERSE_SIZE)


def test_membership_ignores_information_inside_its_own_month(returns, universe):
    """Corrupt returns from month m onwards; month m's membership must not move."""
    months = sorted(universe["month"].unique())
    cut = months[len(months) // 2]

    tampered = returns.copy()
    mask = tampered["date"] >= cut
    tampered.loc[mask, "mktcap_lag"] = (tampered.loc[mask, "mktcap_lag"] * 1000.0).astype("float32")

    after = synthetic.make_universe(tampered, size=UNIVERSE_SIZE)
    before_members = set(universe.loc[universe["month"] == cut, "sec_id"])
    after_members = set(after.loc[after["month"] == cut, "sec_id"])
    assert before_members == after_members


def test_no_company_appears_twice_in_a_month(universe):
    assert not universe.duplicated(subset=["month", "sec_id"]).any()


def test_undersized_universe_is_rejected(universe):
    dropped = universe.drop(universe.index[0]).reset_index(drop=True)
    with pytest.raises(schemas.SchemaError, match="exactly"):
        schemas.validate_universe(dropped, size=UNIVERSE_SIZE)

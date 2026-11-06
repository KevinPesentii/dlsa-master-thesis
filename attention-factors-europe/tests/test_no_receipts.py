"""The receipts experiment (configs/exp_no_receipts_matrix.yaml) claims two universes without depositary
receipts: slots refilled (a receipt is simply not eligible) and slots left empty (ranked with the others,
then left out), and rows that are the US headline with only the data changed."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from afe import schemas
from afe.data import us_panel as up

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_bundle as mb  # noqa: E402
import run_matrix as rm  # noqa: E402
from test_gap2_matrix import changed, flat, load  # noqa: E402
from test_paper_universe import P, _elig  # noqa: E402

CAPS = {11: (6.0e8, "NS"), 12: (5.0e8, "AD"), 13: (4.0e8, "NS"), 14: (3.0e8, "NS"), 15: (2.0e8, "AD"), 16: (1.0e8, "NS")}
SAMPLE = {"start": "2021-01", "end": "2021-01", "universe_size": 3}
CAL = pd.bdate_range("2021-01-04", "2021-01-29")


def lines(keep=("NS", "AD")):
    elig = _elig([dict(permno=p, permco=p, month=P("2020-12"), cap=c, sharetype=t) for p, (c, t) in CAPS.items() if t in keep])
    return up.line_month(elig, up.company_month(elig))


def ranks(uni):
    return uni.set_index("sec_id")["cap_rank"].to_dict()


def test_refilled_universe_takes_the_next_largest_line():
    """Top 3 of the six lines with the receipt 12 in it; without receipts 14 moves up into the third slot."""
    plain, _ = up.build_universe(lines(), {"sample": SAMPLE}, CAL)
    assert ranks(plain) == {"11": 1, "12": 2, "13": 3}
    refilled, _ = up.build_universe(lines(("NS",)), {"sample": SAMPLE}, CAL)
    assert ranks(refilled) == {"11": 1, "13": 2, "14": 3}


def test_left_out_lines_are_not_replaced_and_the_ranks_stay_dense():
    uni, asof = up.build_universe(lines(), {"sample": SAMPLE}, CAL, dropped={P("2020-12"): {12, 15}})
    assert ranks(uni) == {"11": 1, "13": 2}
    assert asof.set_index("primary_permno")["cap_rank_all"].to_dict() == {11: 1, 13: 3}


def test_leaving_out_a_line_below_the_cut_changes_nothing():
    plain, _ = up.build_universe(lines(), {"sample": SAMPLE}, CAL)
    uni, _ = up.build_universe(lines(), {"sample": SAMPLE}, CAL, dropped={P("2020-12"): {15}})
    pd.testing.assert_frame_equal(plain, uni)


def test_the_share_type_is_read_at_the_ranking_date():
    """Membership of January 2021 ranks at the end of December: a flag first seen in January cannot move it."""
    plain, _ = up.build_universe(lines(), {"sample": SAMPLE}, CAL)
    uni, _ = up.build_universe(lines(), {"sample": SAMPLE}, CAL, dropped={P("2021-01"): {12}})
    pd.testing.assert_frame_equal(plain, uni)


def test_validator_takes_any_size_only_when_told_and_still_wants_dense_ranks():
    uni, _ = up.build_universe(lines(), {"sample": SAMPLE}, CAL, dropped={P("2020-12"): {12}})
    schemas.validate_universe(uni, size=None)
    with pytest.raises(schemas.SchemaError, match="exactly 3"):
        schemas.validate_universe(uni, size=3)
    with pytest.raises(schemas.SchemaError, match="cap_rank"):
        schemas.validate_universe(uni.assign(cap_rank=uni["cap_rank"].astype("int16") + 1), size=None)


def test_each_build_differs_from_the_default_only_in_its_one_key_and_outputs():
    base = flat(load("us_data"))
    diff = lambda name: {k for k, v in flat(load(name)).items() if base.get(k) != v} | (base.keys() - flat(load(name)).keys())  # noqa: E731
    assert diff("us_data_no_receipts") == {"eligibility.sharetype", "output.dir", "output.inspect_dir"}
    assert diff("us_data_drop_receipts") == {"universe.drop_after_ranking", "output.dir", "output.inspect_dir"}
    assert load("us_data_no_receipts")["eligibility"]["sharetype"] == ["NS", "UG", "SB", "CE"]
    assert load("us_data_drop_receipts")["universe"]["drop_after_ranking"] == ["AD"]
    dirs = [load(n)["output"][k] for n in ("us_data", "us_data_top800", "us_data_rank301_800", "us_data_no_receipts", "us_data_drop_receipts")
            for k in ("dir", "inspect_dir")]
    assert len(set(dirs)) == len(dirs)


EXPECTED = {"us_headline": {}, "us_drop_receipts": {"data.dir": "data/us/drop_receipts"}, "us_no_receipts": {"data.dir": "data/us/no_receipts"}}


def test_rows_are_the_headline_with_only_the_data_changed():
    matrix, base = load("exp_no_receipts_matrix"), load("us_headline")
    assert {s["name"] for s in matrix["specs"]} == set(EXPECTED)
    for spec in matrix["specs"]:
        assert spec["config"] == "configs/us_headline.yaml" and spec["K"] == [30], spec["name"]
        assert changed(rm.resolve(spec, matrix), base) == {**EXPECTED[spec["name"]], "evaluation.save_models": True}, spec["name"]


def test_matrix_expands_to_five_seeds_of_three_rows(tmp_path):
    matrix = yaml.safe_load((ROOT / "configs" / "exp_no_receipts_matrix.yaml").read_text())
    jobs = rm.expand(matrix, None, tmp_path, threads=2, smoke=False)
    assert sorted(j.id for j in jobs) == sorted(f"{r}_K30_s{s}" for r in EXPECTED for s in range(5))


def test_the_row_joins_the_thesis_matrix_without_its_baseline(tmp_path):
    """Table 11 takes us_no_receipts from its own matrix. That matrix also ran the headline, under the spec the
    thesis matrix uses: the thesis row must keep its own runs, and a spec named twice or not there must fail."""
    from afe.evaluation import thesis_data as td
    runs = tmp_path / "runs"
    for name in ("head_thesis", "head_exp", "refilled"):
        (runs / name).mkdir(parents=True)
        (runs / name / "oos_daily.csv").write_text("date,gross\n")
    for matrix, rows in (("thesis", [("us_attention", "head_thesis")]),
                         ("exp", [("us_attention", "head_exp"), ("us_no_receipts", "refilled")])):
        d = runs / "_matrix" / matrix
        d.mkdir(parents=True)
        pd.DataFrame([{"spec": s, "K": 30, "seed": 0, "SR_net": 0.0, "run_dir": f"/home/ubuntu/dlsa/runs/{r}"}
                      for s, r in rows]).to_csv(d / "summary.csv", index=False)
    M = td.Matrix(runs / "_matrix" / "thesis", runs)
    M.add(runs / "_matrix" / "exp", ["us_no_receipts"])
    assert M.runs("us_no_receipts", 30)["run"].iloc[0].name == "refilled"
    assert [p.name for p in M.runs("us_attention", 30)["run"]] == ["head_thesis"]
    for bad in (["us_no_receipts"], ["us_attention"], ["us_not_there"]):
        with pytest.raises(ValueError):
            M.add(runs / "_matrix" / "exp", bad)


def test_bundle_carries_both_universes():
    for m in ("us_no_receipts", "us_drop_receipts"):
        d = load("exp_no_receipts_matrix")["specs"]
        where = next(s["set"]["data.dir"] for s in d if s["name"] == m.replace("us_", "us_"))
        assert mb.FILES[m] == [f"{where}/{t}.parquet" for t in mb.TABLES]

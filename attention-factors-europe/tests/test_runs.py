import json

from afe import runs


def test_run_directory_records_what_is_needed_to_reproduce(tmp_path):
    run_dir = runs.create_run("smoke", {"market": "us", "K": 30}, seed=7, root=tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())

    assert manifest["seed"] == 7
    assert manifest["config"]["K"] == 30
    assert manifest["git_commit"]
    assert "pandas" in manifest["versions"]

    runs.write_metrics(run_dir, {"sharpe_net": 2.28})
    assert json.loads((run_dir / "metrics.json").read_text())["sharpe_net"] == 2.28


def test_same_name_in_the_same_second_gets_a_new_directory(tmp_path):
    # parallel year processes finish loading at the same moment
    a = runs.create_run("clash", {}, seed=0, root=tmp_path)
    b = runs.create_run("clash", {}, seed=0, root=tmp_path)
    assert a != b
    assert (a / "manifest.json").exists() and (b / "manifest.json").exists()


def test_a_log_cut_off_by_a_hard_stop_loses_only_its_last_record(tmp_path):
    p = tmp_path / "trials.jsonl"
    p.write_text('{"cid": "a", "seed": 0}\n{"cid": "b", "seed": 0}\n{"cid": "c", "se')   # power went mid-write
    assert [r["cid"] for r in runs.read_jsonl(p)] == ["a", "b"]
    with p.open("a") as f:                                   # the resumed run appends on a fresh line
        f.write(json.dumps({"cid": "c", "seed": 0}) + "\n")
    assert [r["cid"] for r in runs.read_jsonl(p)] == ["a", "b", "c"]
    p.write_text('{"cid": "a"}\n{"ci\n{"cid": "b"}\n')         # damage inside the file is not a hard stop
    try:
        runs.read_jsonl(p)
        raise AssertionError("a bad line inside the log must raise")
    except json.JSONDecodeError:
        pass
    assert runs.read_jsonl(tmp_path / "none.jsonl") == []

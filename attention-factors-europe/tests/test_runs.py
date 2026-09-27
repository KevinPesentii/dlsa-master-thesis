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

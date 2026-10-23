"""Check that published examples and experiment claims match their actual files."""
import hashlib
import json
import math
from pathlib import Path

DEMOS = Path(__file__).resolve().parents[1] / "morpholab/static/demos"


def read(name):
    return json.loads((DEMOS / name).read_text())


def test_gallery_has_verified_portable_controllers():
    manifest = read("gallery.json")
    assert manifest["format"] == "rlsim-gallery-v1"
    entries = manifest["entries"]
    assert len({entry["id"] for entry in entries}) == len(entries) >= 3
    for entry in entries:
        path = DEMOS / entry["file"]
        assert path.parent == DEMOS
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        model = read(entry["file"])
        assert model["steps"] == entry["steps"]
        assert model["policy"] and model["normalizer"]
        assert "continuation" not in model  # Native exports are inference-only.
        assert (DEMOS / entry["thumbnail"]).is_file()
        evaluation = entry["browser_evaluation"]
        assert len(evaluation["episodes"]) == len(evaluation["seeds"]) == 3
        if entry["id"].endswith("-walk"):
            assert evaluation["survived"] == 3
            assert evaluation["mean_distance"] > 5


def test_published_comparison_retains_the_complete_protocol():
    report = read("comparison.json")
    protocol = report["protocol"]
    assert report["completed_at"]
    expected = {(body, seed, condition) for body in protocol["bodies"]
                for seed in protocol["training_seeds"] for condition in protocol["conditions"]}
    assert len(expected) == len(report["trials"]) == 30
    assert {(t["body"], t["seed"], t["condition"]) for t in report["trials"]} == expected
    for trial in report["trials"]:
        assert trial["status"] in ["completed", "failed"]
        assert trial["budget"] == trial["config"]["total_steps"] == protocol["steps"]
        if trial["status"] == "failed":
            assert trial["error"]
            continue
        assert trial["steps"] == protocol["steps"]
        assert trial["checkpoint_steps"] <= trial["steps"]
        selection_seeds = {(trial["seed"] + 1000003 + i * 997) % 2147483647
                           for i in range(trial["config"]["evaluation_episodes"])}
        assert not selection_seeds.intersection(protocol["evaluation_seeds"])
        assert [e["seed"] for e in trial["episodes"]] == protocol["evaluation_seeds"]
        for key in ["speed_error", "distance", "seconds", "return"]:
            mean = sum(e[key] for e in trial["episodes"]) / len(trial["episodes"])
            assert math.isclose(mean, trial["metrics"][key])

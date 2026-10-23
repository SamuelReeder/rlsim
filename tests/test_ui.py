"""Opt-in browser checks against a running app, without launching training.

    MORPHOLAB_TEST_URL=http://localhost:8777 .venv/bin/pytest -q tests/test_ui.py

Requires the development Playwright dependency and its Chromium browser.
Inference uses an existing checkpoint; all training requests are intercepted.
"""

import hashlib
import json
import os
from pathlib import Path
import re

import pytest


BASE_URL = os.environ.get("MORPHOLAB_TEST_URL", "").rstrip("/")
pytestmark = pytest.mark.skipif(
    not BASE_URL, reason="Set MORPHOLAB_TEST_URL to check a running app"
)
ARTIFACTS = Path(__file__).resolve().parents[1] / "test-results" / "redesign"
ADVANCED_FIELDS = {
    "hidden-size": "hidden_size", "hidden-layers": "hidden_layers",
    "learning-rate": "learning_rate", "batch-size": "batch_size",
    "epochs": "epochs", "rollout-steps": "rollout_steps", "discount": "gamma",
    "entropy-coef": "entropy_coef", "clip-range": "clip_range",
}
SERVER_ADVANCED = {
    "hidden_size": 256, "hidden_layers": 2, "learning_rate": .0003,
    "batch_size": 512, "epochs": 10, "rollout_steps": 256,
    "gamma": .99, "entropy_coef": .005, "clip_range": .2,
}
CUSTOM_ADVANCED = {
    "hidden_size": 48, "hidden_layers": 3, "learning_rate": .0007,
    "batch_size": 96, "epochs": 3, "rollout_steps": 64,
    "gamma": .95, "entropy_coef": .01, "clip_range": .15,
}


@pytest.fixture(scope="module")
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def training_requests():
    return []


@pytest.fixture
def page(browser, training_requests):
    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def intercept_training(route):
        training_requests.append(route.request.post_data_json)
        route.fulfill(
            status=409,
            content_type="application/json",
            body='{"detail":"Training request intercepted by browser validation."}',
        )

    # A browser test must never start an expensive training job accidentally.
    page.route("**/api/train", intercept_training)
    page.goto(f"{BASE_URL}/?engine=server")
    page.wait_for_function(
        "() => document.querySelector('[data-tab=\"body\"]')"
        " && document.querySelector('#arena-empty')?.hidden"
        " && !document.querySelector('#baseline-button')?.disabled"
    )
    yield page
    context.close()
    assert not errors, f"Uncaught browser errors: {errors}"


def select_tab(page, name):
    tab = page.locator(f'[data-tab="{name}"]')
    tab.click()
    assert tab.get_attribute("aria-selected") == "true"
    panel_id = tab.get_attribute("aria-controls")
    assert panel_id, "Tabs must expose their associated panel"
    assert page.locator(f"#{panel_id}").is_visible()
    page.wait_for_function("() => !document.querySelector('#baseline-button').disabled")
    return page.locator(f"#{panel_id}")


def edit_number(page, selector, value):
    field = page.locator(selector)
    field.fill(str(value))
    field.dispatch_event("input")
    return field


def preview_after_edit(page, selector, value):
    with page.expect_response(
        lambda response: response.url.endswith("/api/preview")
        and response.request.method == "POST"
    ) as response:
        edit_number(page, selector, value)
    assert response.value.status == 200, response.value.text()
    return response.value.json()


@pytest.mark.parametrize("width,height", [(1440, 1000), (390, 844)])
def test_minimal_layout_and_tabs(page, width, height):
    page.set_viewport_size({"width": width, "height": height})
    assert page.locator("[role=tab]").count() == 4
    assert page.title() == "RLsim"
    assert page.locator("h1").inner_text() == "RLsim"
    assert page.locator(".page-intro, .research-panel, .page-footer").count() == 0
    assert page.locator('a[href*="research"]').count() == 0
    body_text = page.locator("body").inner_text().lower()
    for removed_copy in (
        "first, a body",
        "then, a behavior",
        "intelligence, one footstep",
        "experience becomes progress",
        "from research to movement",
    ):
        assert removed_copy not in body_text
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for tab in ("body", "train", "run", "explore"):
        select_tab(page, tab)
        assert page.locator("[role=tabpanel]:visible").count() == 1
        assert page.locator("#simulation").is_visible()
        assert page.evaluate(
            "() => document.documentElement.scrollWidth <= innerWidth + 1"
        ), f"{tab} overflows horizontally at {width}px"
        page.screenshot(path=str(ARTIFACTS / f"{tab}-{width}.png"), full_page=True)


def test_body_and_environment_edits_reach_simulator(page):
    select_tab(page, "body")
    for preset, legs in (("biped", 2), ("hexapod", 6), ("quadruped", 4)):
        with page.expect_response("**/api/preview") as response:
            page.locator("#body-preset").select_option(preset)
        result = response.value.json()
        assert result["body"]["legs"] == legs
        assert result["action_size"] == legs * 2

    result = preview_after_edit(page, '[data-field="torso_length"]', 1.15)
    assert result["body"]["torso_length"] == 1.15
    torso = next(geom for geom in result["frame"]["geoms"] if geom["type"] == "capsule")
    assert torso["size"][1] == pytest.approx(1.15 / 2)

    result = preview_after_edit(page, "#gravity", 4.9)
    assert result["environment"]["gravity"] == 4.9
    result = preview_after_edit(page, "#friction", 0.45)
    assert result["environment"]["friction"] == 0.45


def test_body_reset_baseline_and_load_model_settings(page):
    select_tab(page, "body")
    preview_after_edit(page, '[data-field="torso_length"]', 1.15)
    with page.expect_response("**/api/preview") as response:
        page.locator("#reset-design").click()
    default = page.request.get(f"{BASE_URL}/api/config").json()["presets"]["quadruped"]
    assert response.value.json()["body"] == default
    preview_after_edit(page, "#gravity", 4.9)
    preview_after_edit(page, "#friction", 0.45)
    with page.expect_response("**/api/replay") as response:
        page.locator("#baseline-button").click()
    result = response.value.json()
    assert result["source"] == "untrained"
    assert result["body"] == default
    assert result["environment"] == {"gravity": 4.9, "friction": 0.45}
    assert result["summary"]["mean_squared_action"] == 0
    assert result["frames"][-1]["time"] > 0
    assert page.locator("#viewer-mode").inner_text() == "Zero torque"

    select_tab(page, "body")
    preview_after_edit(page, '[data-field="torso_length"]', 1.15)
    select_tab(page, "run")
    selected_id = page.locator("#saved-run").input_value()
    if not selected_id:
        pytest.skip("No saved model is available for restoring settings")
    runs = page.request.get(f"{BASE_URL}/api/runs").json()["runs"]
    config = next(run["config"] for run in runs if run["id"] == selected_id)
    with page.expect_response("**/api/preview"):
        page.locator("#load-run").click()
    select_tab(page, "body")
    for field, value in config["body"].items():
        if field in {"name", "legs"}:
            continue
        assert float(page.locator(f'[data-field="{field}"]').input_value()) == value
    env = config.get("environment", {"gravity": 9.81, "friction": 1})
    assert float(page.locator("#gravity").input_value()) == env["gravity"]
    assert float(page.locator("#friction").input_value()) == env["friction"]


def test_training_controls_build_requested_job_without_starting_it(page, training_requests):
    runs_before = page.request.get(f"{BASE_URL}/api/runs").json()["runs"]
    select_tab(page, "body")
    preview_after_edit(page, '[data-field="upper_length"]', 0.55)
    preview_after_edit(page, "#gravity", 8.3)
    preview_after_edit(page, "#friction", 0.7)
    select_tab(page, "train")
    edit_number(page, "#training-budget", 8192)
    edit_number(page, "#target-speed", 1.4)
    edit_number(page, "#seed", 19)
    edit_number(page, "#n-envs", 4)
    page.locator("#curriculum").uncheck()
    page.locator("#randomize").uncheck()
    page.locator("#advanced-training summary").click()
    for field, key in ADVANCED_FIELDS.items():
        edit_number(page, f"#{field}", CUSTOM_ADVANCED[key])
    with page.expect_response("**/api/train"):
        page.locator("#train-button").click()
    assert len(training_requests) == 1
    payload = training_requests[0]
    assert payload["body"]["upper_length"] == 0.55
    assert payload["environment"] == {"gravity": 8.3, "friction": 0.7}
    assert payload["total_steps"] == 8192
    assert payload["target_speed"] == 1.4
    assert payload["seed"] == 19
    assert payload["n_envs"] == 4
    assert payload["curriculum"] is False
    assert payload["randomize"] is False
    assert payload["device"] == "auto"
    assert payload["task"] is None
    for key, value in CUSTOM_ADVANCED.items():
        assert payload[key] == value
    runs_after = page.request.get(f"{BASE_URL}/api/runs").json()["runs"]
    assert [run["id"] for run in runs_after] == [run["id"] for run in runs_before]


def test_advanced_defaults_reset_and_collapsed_validation(page, training_requests):
    select_tab(page, "train")
    details = page.locator("#advanced-training")
    assert not details.evaluate("el => el.open")
    details.locator("summary").click()
    for field, key in ADVANCED_FIELDS.items():
        assert float(page.locator(f"#{field}").input_value()) == SERVER_ADVANCED[key]
        edit_number(page, f"#{field}", CUSTOM_ADVANCED[key])
    page.locator("#reset-advanced").click()
    for field, key in ADVANCED_FIELDS.items():
        assert float(page.locator(f"#{field}").input_value()) == SERVER_ADVANCED[key]
    # A hidden invalid value must open the section and focus its correction,
    # without sending a training request.
    for field, invalid in (("hidden-size", 257), ("hidden-layers", 1.5),
                           ("learning-rate", 0), ("batch-size", "")):
        page.locator(f"#{field}").fill(str(invalid))
        details.locator("summary").click()
        assert not details.evaluate("el => el.open")
        page.locator("#train-button").click()
        assert details.evaluate("el => el.open")
        assert page.evaluate("document.activeElement.id") == field
        assert not training_requests
        page.locator("#reset-advanced").click()


@pytest.mark.parametrize("legacy", [False, True])
def test_model_settings_restore_advanced_values_or_legacy_defaults(page, legacy):
    source = page.request.get(f"{BASE_URL}/api/config").json()
    config = {
        **source["defaults"], "body": source["presets"]["biped"],
        "environment": source["environment_defaults"],
    }
    for key in ADVANCED_FIELDS.values():
        config.pop(key, None)
    if not legacy:
        config.update(CUSTOM_ADVANCED)
    run = {
        "id": "20261020T160000-abcdabcd", "status": "completed", "steps": 4096,
        "created_at": "2026-10-20T16:00:00Z", "checkpoint_available": True,
        "name": config["body"]["name"], "config": config,
    }
    page.route("**/api/runs", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps({"runs": [run]}),
    ))
    page.route("**/api/status", lambda route: route.fulfill(
        status=200, content_type="application/json",
        body=json.dumps({"status": "idle", "history": [], "metrics": {}}),
    ))
    page.reload()
    page.wait_for_function("() => !document.querySelector('#baseline-button').disabled")
    select_tab(page, "train")
    page.locator("#advanced-training summary").click()
    edit_number(page, "#hidden-size", 17)
    edit_number(page, "#learning-rate", .002)
    select_tab(page, "run")
    with page.expect_response("**/api/preview"):
        page.locator("#load-run").click()
    select_tab(page, "train")
    expected = SERVER_ADVANCED if legacy else CUSTOM_ADVANCED
    for field, key in ADVANCED_FIELDS.items():
        assert float(page.locator(f"#{field}").input_value()) == expected[key]


def test_active_run_restores_and_locks_advanced_controls(page):
    source = page.request.get(f"{BASE_URL}/api/config").json()
    task = {
        "preset": "target", "episode_seconds": 15, "target": [3, -1, .9],
        "tracked_body": "", "terminate_on_fall": False,
        "min_height_ratio": .2, "max_tilt_degrees": 85,
    }
    config = {**source["defaults"], **CUSTOM_ADVANCED, "body": source["presets"]["biped"], "task": task}
    status = {
        "status": "training", "run_id": "20261020T160000-abcdabcd",
        "config": config, "history": [], "metrics": {}, "steps": 1024,
        "total_steps": 4096, "elapsed": 1, "checkpoint_available": False,
        "hardware": source["hardware"],
    }
    page.route("**/api/status", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(status),
    ))
    page.reload()
    page.wait_for_function("() => document.querySelector('#run-status').textContent === 'Training'")
    page.locator("[data-tab=train]").click()
    page.locator("#advanced-training summary").click()
    for field, key in ADVANCED_FIELDS.items():
        assert page.locator(f"#{field}").is_disabled()
        assert float(page.locator(f"#{field}").input_value()) == CUSTOM_ADVANCED[key]
    assert page.locator("#reset-advanced").is_disabled()
    assert page.locator("#import-body").is_disabled()
    assert page.locator("#export-body").is_disabled()
    page.locator("#task-settings > summary").click()
    assert page.locator("#task-preset").input_value() == "target"
    assert float(page.locator("#target-x").input_value()) == 3
    for field in ("task-preset", "episode-seconds", "tracked-body", "target-x",
                  "target-y", "target-z", "terminate-on-fall", "min-height-ratio",
                  "max-tilt", "reward-expression", "reset-task"):
        assert page.locator(f"#{field}").is_disabled()


def test_task_controls_build_custom_task_and_reset_legacy_default(page, training_requests):
    select_tab(page, "train")
    page.locator("#task-settings > summary").click()
    assert page.locator("#task-target-fields").is_hidden()
    assert page.locator("#custom-reward-field").is_hidden()
    page.locator("#task-preset").select_option("custom")
    for field, value in {
        "episode-seconds": 15, "target-x": 3, "target-y": -1, "target-z": .9,
        "min-height-ratio": .2, "max-tilt": 85,
    }.items():
        edit_number(page, f"#{field}", value)
    page.locator("#terminate-on-fall").uncheck()
    page.locator("#reward-expression").fill("speed_tracking + 0.2*upright")
    with page.expect_response("**/api/train"):
        page.locator("#train-button").click()
    assert training_requests[-1]["task"] == {
        "preset": "custom", "episode_seconds": 15, "target": [3, -1, .9],
        "tracked_body": "", "terminate_on_fall": False,
        "min_height_ratio": .2, "max_tilt_degrees": 85,
        "expression": "speed_tracking + 0.2*upright",
    }
    page.locator("#reward-expression").fill("")
    page.locator("#task-settings > summary").click()
    page.locator("#train-button").click()
    assert page.locator("#task-settings").evaluate("el => el.open")
    assert page.evaluate("document.activeElement.id") == "reward-expression"
    assert len(training_requests) == 1
    page.locator("#reset-task").click()
    assert page.locator("#task-preset").input_value() == "velocity"
    assert page.locator("#task-target-fields").is_hidden()
    assert page.locator("#custom-reward-field").is_hidden()
    with page.expect_response("**/api/train"):
        page.locator("#train-button").click()
    assert training_requests[-1]["task"] is None


def test_task_preview_updates_and_body_import_clears_absent_tracked_body(page):
    select_tab(page, "train")
    page.locator("#task-settings > summary").click()
    page.locator("#task-preset").select_option("target")
    result = preview_after_edit(page, "#target-x", 4.5)
    assert result["frame"]["target"] == [4.5, 0, .8]
    page.locator("#tracked-body").select_option("lower_3")
    page.wait_for_function("() => !document.querySelector('#baseline-button').disabled")
    select_tab(page, "body")
    biped = page.request.get(f"{BASE_URL}/api/config").json()["presets"]["biped"]
    with page.expect_response("**/api/preview"):
        page.locator("#body-file").set_input_files({
            "name": "biped.json", "mimeType": "application/json",
            "buffer": json.dumps(biped).encode(),
        })
    page.wait_for_function("() => document.querySelector('#status-text').textContent === 'Body imported'")
    assert page.locator("#tracked-body").input_value() == ""
    assert "lower_3" not in page.locator("#tracked-body option").evaluate_all(
        "options => options.map(option => option.value)"
    )
    select_tab(page, "train")
    assert page.locator("#task-preset").input_value() == "target"
    assert float(page.locator("#target-x").input_value()) == 4.5


def test_model_import_shows_only_options_used_by_selected_format(page):
    def force_cpu(route):
        response = route.fetch()
        route.fulfill(response=response, body=(
            "Object.defineProperty(navigator,'gpu',{value:undefined});\n" + response.text()
        ))
    page.route("**/browser-worker.js", force_cpu)
    page.locator("#body-import > summary").click()
    page.locator("#asset-files").set_input_files([
        {"name": "robot.xml", "mimeType": "application/xml", "buffer": b"<mujoco/>"},
        {"name": "character.glb", "mimeType": "model/gltf-binary", "buffer": b"glTF"},
    ])
    page.wait_for_function(
        "() => !document.querySelector('#body-import-options').hidden"
        " && !document.querySelector('#import-entry').disabled", timeout=60000,
    )
    assert page.locator("#execution-engine").input_value() == "browser"
    initially_visual = page.locator("#import-entry").input_value().endswith(".glb")
    assert page.locator("#import-scale").is_visible() is initially_visual
    assert page.locator("#import-floating-base").is_visible() is (not initially_visual)
    for filename, visual in (("robot.xml", False), ("character.glb", True), ("robot.xml", False)):
        page.locator("#import-entry").select_option(filename)
        for field in ("import-scale", "import-mass", "import-rig"):
            assert page.locator(f"#{field}").is_visible() is visual
        for field in ("import-floating-base", "import-auto-actuators", "import-add-floor"):
            assert page.locator(f"#{field}").is_visible() is (not visual)
        assert page.locator("#import-motor-strength").is_visible()
    label = page.locator("label", has=page.locator("#import-auto-actuators"))
    assert label.inner_text() == "Generate actuators if none"
    assert "Authored actuators are retained" in label.get_attribute("title")


@pytest.mark.parametrize("weights,expression", [
    ({"alive": .7, "energy": .02, "fall": 2}, "0.7*alive + 0.02*energy + 2*fall"),
    ({}, "0"),
])
def test_saved_reward_weights_restore_as_visible_custom_expression(page, training_requests, weights, expression):
    source = page.request.get(f"{BASE_URL}/api/config").json()
    task = {
        "preset": "balance", "weights": weights, "target": [4, 1, .5],
        "episode_seconds": 10, "terminate_on_fall": False,
    }
    config = {
        **source["defaults"], "body": source["presets"]["biped"],
        "environment": source["environment_defaults"], "task": task,
    }
    run = {
        "id": "20261020T160000-abcdabcd", "status": "completed", "steps": 4096,
        "created_at": "2026-10-20T16:00:00Z", "checkpoint_available": True,
        "config": config,
    }
    page.route("**/api/runs", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps({"runs": [run]}),
    ))
    page.route("**/api/status", lambda route: route.fulfill(
        status=200, content_type="application/json",
        body=json.dumps({"status": "idle", "history": [], "metrics": {}}),
    ))
    page.reload()
    page.wait_for_function("() => !document.querySelector('#baseline-button').disabled")
    select_tab(page, "run")
    with page.expect_response("**/api/preview"):
        page.locator("#load-run").click()
    select_tab(page, "train")
    page.locator("#task-settings > summary").click()
    assert page.locator("#task-preset").input_value() == "custom"
    assert page.locator("#reward-expression").input_value() == expression
    with page.expect_response("**/api/train"):
        page.locator("#train-button").click()
    restored = training_requests[-1]["task"]
    assert restored["preset"] == "custom"
    assert restored["expression"] == expression
    assert restored["target"] == task["target"]
    assert restored["episode_seconds"] == 10
    assert restored["terminate_on_fall"] is False


def test_saved_policy_inference_and_playback_preserve_checkpoint(page):
    runs = page.request.get(f"{BASE_URL}/api/runs").json()["runs"]
    checkpoints = [run for run in runs if run["checkpoint_available"]]
    if not checkpoints:
        pytest.skip("No saved policy is available for inference")
    selected = max(checkpoints, key=lambda run: run["steps"])
    checkpoint_url = f"{BASE_URL}/api/checkpoint?run_id={selected['id']}"
    before = page.request.get(checkpoint_url)
    assert before.ok
    checkpoint_hash = hashlib.sha256(before.body()).hexdigest()

    select_tab(page, "run")
    saved_ids = page.locator("#saved-run option").evaluate_all(
        "options => options.map(option => option.value).filter(Boolean)"
    )
    assert set(saved_ids) == {run["id"] for run in checkpoints}
    page.locator("#saved-run").select_option(selected["id"])
    select_tab(page, "body")
    preview_after_edit(page, "#gravity", 8.5)
    preview_after_edit(page, "#friction", 0.85)
    select_tab(page, "run")
    edit_number(page, "#eval-seed", 237)
    edit_number(page, "#eval-speed", 1.2)
    with page.expect_response("**/api/replay", timeout=60000) as response:
        page.locator("#replay-button").click()
    assert response.value.status == 200, response.value.text()
    payload = response.value.request.post_data_json
    result = response.value.json()
    assert payload["run_id"] == selected["id"]
    assert result["source"] == "trained"
    assert result["body"] == selected["config"]["body"]
    assert result["environment"] == {"gravity": 8.5, "friction": 0.85}
    assert result["target_speed"] == 1.2
    assert result["seed"] == 237
    assert result["summary"]["steps"] > 0
    assert len(result["frames"]) > 1

    pause = page.locator("#play-pause")
    page.wait_for_function("() => !document.querySelector('#play-pause').disabled")
    if pause.inner_text() == "Pause":
        pause.click()
    assert pause.inner_text() == "Play"
    page.locator("#playback-speed").select_option("2")
    assert page.locator("#playback-speed").input_value() == "2"
    page.locator("#timeline").evaluate(
        """el => {
            el.value = (Number(el.min) + Number(el.max)) / 2;
            el.dispatchEvent(new Event('input', {bubbles: true}));
        }"""
    )
    assert float(re.search(r"[\d.]+", page.locator("#sim-time").inner_text()).group()) > 0
    paused_time = page.locator("#sim-time").inner_text()
    page.wait_for_timeout(250)
    assert page.locator("#sim-time").inner_text() == paused_time
    page.locator("#view-angle").select_option("side")
    assert page.locator("#view-angle").input_value() == "side"
    page.locator("#reset-camera").click()
    assert page.locator("#view-angle").input_value() == "perspective"
    page.screenshot(path=str(ARTIFACTS / "inference-desktop.png"), full_page=True)

    with page.expect_download() as model_download:
        page.locator("#checkpoint-link").click()
    assert model_download.value.suggested_filename.endswith(".zip")
    assert hashlib.sha256(Path(model_download.value.path()).read_bytes()).hexdigest() == checkpoint_hash
    with page.expect_download() as body_download:
        page.locator("#body-link").click()
    assert body_download.value.suggested_filename == "body.xml"
    assert "<mujoco" in Path(body_download.value.path()).read_text()

    after = page.request.get(checkpoint_url)
    assert after.ok
    assert hashlib.sha256(after.body()).hexdigest() == checkpoint_hash

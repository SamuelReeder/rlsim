"""Opt-in physics editing checks; static hosting, isolated browser, no RL job.

MORPHOLAB_BROWSER_TEST=1 .venv/bin/pytest -q tests/test_rig_ui.py
"""
import json
import os
from pathlib import Path
import subprocess

import pytest
from test_browser_local import ROOT, static_url

pytestmark = pytest.mark.skipif(
    not os.getenv("MORPHOLAB_BROWSER_TEST"), reason="Opt-in real browser rig editor"
)


@pytest.fixture
def page(static_url):
    pw = pytest.importorskip("playwright.sync_api")
    with pw.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000})

        def no_gpu(route):
            response = route.fetch()
            route.fulfill(response=response, body="Object.defineProperty(navigator,'gpu',{value:undefined});\n" + response.text())

        context.route("**/browser-worker.js", no_gpu)
        page = context.new_page()
        errors, api_requests = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: api_requests.append(request.url) if "/api/" in request.url else None)
        page.goto(static_url)
        page.wait_for_function("document.querySelector('#arena-empty').hidden", timeout=90000)
        yield page
        context.close()
        browser.close()
        assert not errors, errors
        assert not api_requests, api_requests


def import_visual(page, path):
    page.locator("#asset-files").set_input_files(str(path))
    page.wait_for_function("!document.querySelector('#body-import-options').hidden")
    assert page.locator("#import-profile").is_visible()
    page.locator("#apply-body-import").click()
    page.wait_for_function("document.querySelector('#status-text').textContent === 'Body imported on This device'", timeout=60000)
    assert page.locator("#physical-rig-report").is_visible()
    assert not page.locator("#physical-rig-editor").evaluate("el=>el.open")


def test_physics_editor_commits_valid_changes_and_keeps_invalid_draft(page, tmp_path):
    fixture = tmp_path / "character.glb"
    subprocess.run(["node", "--input-type=module", "-e",
                    "import {tinyRigGLB} from './tests/scene-fixtures.mjs'; import {writeFileSync} from 'node:fs'; writeFileSync(process.argv[1],new Uint8Array(tinyRigGLB()));",
                    str(fixture)], cwd=ROOT, check=True)
    import_visual(page, fixture)
    original = page.evaluate("structuredClone(state.body)")
    page.evaluate("""() => {
      const saved = {id: 'previous-controller', config: {body: structuredClone(state.body)}};
      state.runs.push(saved);
      state.selectedRun=saved.id;
      document.querySelector('#saved-run').add(new Option('Previous controller',saved.id));
      document.querySelector('#saved-run').value=saved.id;
      controls();
    }""")
    page.locator("#physical-rig-editor > summary").click()
    assert page.locator("#rig-parent").is_disabled()
    assert page.locator("#rig-joint").is_disabled()
    root_index = next(index for index, part in enumerate(original["physical_rig"]["bodies"]) if part["id"] == original["physical_rig"]["root"])
    mass = original["physical_rig"]["bodies"][root_index]["mass"]
    page.locator("#rig-mass").fill(str(mass * 2))
    assert page.locator("#train-button").is_disabled()
    assert page.locator("#export-body").is_disabled()
    assert page.evaluate(f"state.body.physical_rig.bodies[{root_index}].mass") == mass
    page.locator("#apply-physical-rig").click()
    page.wait_for_function("document.querySelector('#status-text').textContent.startsWith('Physics applied')", timeout=30000)
    revised = page.evaluate("structuredClone(state.body)")
    assert revised["physical_rig"]["bodies"][root_index]["mass"] == pytest.approx(mass * 2)
    assert revised["xml"] != original["xml"]
    assert revised["visual"]["files"] == original["visual"]["files"]
    assert not page.evaluate("state.selectedRun")
    assert page.evaluate("state.runs.find(run=>run.id==='previous-controller').config.body.xml") == original["xml"]
    assert not page.locator("#train-button").is_disabled()

    # Invalid joint limits must not replace the body or the last valid preview.
    part = next(part for part in revised["physical_rig"]["bodies"] if part["joint"]["type"] == "ball")
    page.locator("#rig-body").select_option(part["id"])
    page.locator("#rig-limit-max").fill("-1")
    page.locator("#apply-physical-rig").click()
    page.wait_for_function("document.querySelector('#rig-draft-message').classList.contains('error')")
    assert page.evaluate("state.body.xml") == revised["xml"]
    assert page.locator("#train-button").is_disabled()
    page.locator("#discard-physical-rig").click()
    assert not page.locator("#train-button").is_disabled()

    # External graph edits are rebuilt when imported, rather than leaving stale XML.
    revised["physical_rig"]["bodies"][root_index]["mass"] = mass * 3
    exported = tmp_path / "edited-body.json"
    exported.write_text(json.dumps({"format": "rlsim-body-v1", "body": revised}))
    page.locator("#body-file").set_input_files(str(exported))
    page.wait_for_function("document.querySelector('#status-text').textContent === 'Body imported'", timeout=30000)
    assert page.evaluate(f"state.body.physical_rig.bodies[{root_index}].mass") == pytest.approx(mass * 3)
    assert page.evaluate("state.body.xml") != revised["xml"]


def test_authored_physics_and_optional_generator_help(page):
    page.locator("#asset-files").set_input_files(str(ROOT / "examples/imports/free-body.xml"))
    page.wait_for_function("!document.querySelector('#body-import-options').hidden")
    assert page.locator("#import-profile").is_hidden()
    page.locator("#apply-body-import").click()
    page.wait_for_function("document.querySelector('#status-text').textContent === 'Body imported on This device'", timeout=60000)
    assert page.locator("#physical-rig-report").is_hidden()
    assert page.locator("#authored-physics-note").is_visible()
    assert not page.locator("#optional-rigging").evaluate("el=>el.open")
    page.locator("#optional-rigging > summary").click()
    assert "14 GB" in page.locator("#optional-rigging").inner_text()
    link = page.locator("#optional-rigging a").get_attribute("href")
    result = page.request.get(page.url + link.removeprefix("./").split("#")[0])
    assert result.ok
    assert "rlsim rig-prepare" in result.text()
    assert "--doctor" in result.text()


def test_female_import_layout_screenshots(page):
    fixture = os.environ.get("RLSIM_FEMALE_FIXTURE")
    female = Path(fixture) if fixture else None
    if not female or not female.exists():
        pytest.skip("Set RLSIM_FEMALE_FIXTURE to a rigged character .glb to run this check")
    import_visual(page, female)
    page.locator("#show-colliders").check()
    page.locator("#physical-rig-editor > summary").click()
    part = page.evaluate("state.rigDraft.bodies.find(p=>p.joint.type==='hinge')?.id")
    if part:
        page.locator("#rig-body").select_option(part)
    output = ROOT / "test-results" / "rig-ui"
    output.mkdir(parents=True, exist_ok=True)
    for width, height in [(1440, 1000), (900, 700), (390, 844)]:
        page.set_viewport_size({"width": width, "height": height})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth+1")
        page.locator("#physical-rig-editor > summary").scroll_into_view_if_needed()
        page.screenshot(path=str(output / f"female-rig-{width}.png"), full_page=True)

"""The published generalization claims must agree with the complete experiment."""
import hashlib
import json
import math
from pathlib import Path
import pytest
from morpholab.morphology import region

ROOT=Path(__file__).resolve().parents[1]/'morpholab/static/demos'


def test_generalization_report_and_selected_controllers():
    report=json.loads((ROOT/'generalization-report.json').read_text())
    manifest=json.loads((ROOT/'generalization.json').read_text())
    protocol=report['protocol']
    assert report['completed_at']
    assert len(report['trials'])==len(protocol['training_seeds'])*2==6
    assert {(t['seed'],t['condition']) for t in report['trials']}=={
        (seed,condition) for seed in protocol['training_seeds'] for condition in protocol['conditions']}
    for trial in report['trials']:
        assert trial['status'] in ['completed','failed']
        if trial['status']=='failed':
            assert trial['error'];continue
        assert trial['steps']==protocol['steps']
        assert len(trial['cases'])==len(protocol['body_cases'])==12
        assert trial['checkpoint_steps']<=trial['steps']
        validation_seeds={(trial['seed']+1000003+i*997)%2147483647 for i in range(3)}
        assert not validation_seeds.intersection(protocol['evaluation_seeds'])
        for case in trial['cases']:
            assert case['region']==region(case['scales'],protocol['width'])
            assert [e['seed'] for e in case['episodes']]==protocol['evaluation_seeds']
            expected=sum(not e['fell'] and e['seconds']>=19.99 and e['speed_error']<=.35 for e in case['episodes'])/3
            assert math.isclose(expected,case['success_rate'])
    for key,condition in [('varied','varied bodies'),('fixed','fixed body')]:
        entry=manifest['controllers'][key]
        chosen=max((t for t in report['trials'] if t['status']=='completed' and t['condition']==condition),key=lambda t:t['validation']['mean_return'])
        assert entry['run_id']==chosen['run_id']
        path=ROOT/entry['file'];assert path.parent==ROOT
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        record=json.loads(path.read_text())
        assert record['steps']==chosen['checkpoint_steps']
        assert record['config']['morphology']['enabled']==(key=='varied')
        assert 'continuation' not in record
        fingerprint=hashlib.sha256(json.dumps({'weights':record['policy']['weights'],'normalizer':record['normalizer']},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert entry['fingerprint']==fingerprint


def test_browser_checks_keep_all_bodies_and_match_bundled_controllers():
    check=json.loads((ROOT/'generalization-browser.json').read_text())
    report=json.loads((ROOT/'generalization-report.json').read_text())
    manifest=json.loads((ROOT/'generalization.json').read_text())
    assert check['controllers']==manifest['controllers']
    assert len(check['cases'])==24
    assert {(c['controller'],c['name']) for c in check['cases']}=={
        (controller,case['name']) for controller in ['varied','fixed'] for case in report['protocol']['body_cases']}
    for case in check['cases']:
        assert len(case['episodes'])==3
        expected=sum(not e['fell'] and e['seconds']>=19.99 and e['speed_error']<=.35 for e in case['episodes'])/3
        assert math.isclose(case['success_rate'],expected)

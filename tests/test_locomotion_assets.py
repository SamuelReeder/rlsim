"""Published broad-coverage claims must match frozen, fully audited policies."""
import hashlib,json
from pathlib import Path
from morpholab.motion import walking_pass
ROOT=Path(__file__).resolve().parents[1]/'morpholab/static/demos'


def test_broad_policy_and_complete_independent_audits():
    r=json.loads((ROOT/'locomotion-study.json').read_text());assert r['completed_at']
    c=r['controller'];path=ROOT/c['file'];assert path.parent==ROOT
    assert hashlib.sha256(path.read_bytes()).hexdigest()==c['sha256']
    record=json.loads(path.read_text())
    assert record['steps']==c['steps']<=c['training_steps']
    assert record['config']['morphology']==c['morphology']
    assert c['morphology']['width']==.4 and c['morphology']['withhold_width']==.2
    assert record['config']['task']['foot_only_support'] is True
    fingerprint=hashlib.sha256(json.dumps({'weights':record['policy']['weights'],'normalizer':record['normalizer']},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert fingerprint==c['fingerprint']
    native=r['native_final'];assert native['suite']=='final' and native['completed']
    development={e['seed'] for audit in r['condition_validation'] for case in audit['cases'] for e in case['episodes']}
    assert not development.intersection(native['seeds'])
    assert not set(e['seed'] for e in c['validation']['episodes']).intersection(native['seeds'])
    assert r['boundary_audit']['completed'] and len(r['boundary_audit']['cases'])==43
    assert r['selection']['checkpoint_sha256']==native['checkpoint_sha256']
    audits=[r['baseline_final'],native,*r['browser_final']]
    for audit,row in zip(audits,r['comparison_rows']):
        assert len(audit['cases'])==29 and len(audit['seeds'])==3
        assert [case['scales'] for case in audit['cases']]==[case['scales'] for case in native['cases']]
        episodes=[e for case in audit['cases'] for e in case['episodes']]
        assert len(episodes)==87
        assert all(e['gait_passed']==walking_pass(e) for e in episodes)
        assert row[2]==f'{sum(e["gait_passed"] for e in episodes)}/{len(episodes)}'
        assert all(e['effort']['absolute_motor_work_j']>=0 for e in episodes)
    for browser in r['browser_final']:
        assert browser['completed_at'] and browser['weights_unchanged']
        assert browser['record_sha256']==c['sha256']
    assert {x['backend'] for x in r['browser_final']}=={'browser CPU','browser WebGPU'}

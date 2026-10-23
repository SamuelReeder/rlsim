"""Keep published gait claims tied to frozen weights and measured episodes."""
import hashlib,json
from pathlib import Path
from morpholab.motion import walking_pass
ROOT=Path(__file__).resolve().parents[1]/'morpholab/static/demos'


def test_walking_report_matches_checkpoints_and_measured_gaits():
    report=json.loads((ROOT/'walking-study.json').read_text())
    assert report['completed_at']
    assert len(report['evaluation_seeds'])==3
    assert len(report['cases'])==39
    for key,entry in report['controllers'].items():
        path=ROOT/entry['file'];assert path.parent==ROOT
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        record=json.loads(path.read_text())
        assert record['config']['task']['preset']=='walk'
        assert record['model_info']['action_schema']=='rlsim-joint-targets-v1'
        assert record['policy']['obsSize']==record['normalizer']['size']==57
        assert record['steps']==entry['steps']<=entry['training_steps']
        fingerprint=hashlib.sha256(json.dumps({'weights':record['policy']['weights'],'normalizer':record['normalizer']},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert fingerprint==entry['fingerprint']
        assert not set(report['evaluation_seeds']).intersection(e['seed'] for e in entry['validation']['episodes'])
    for case in report['cases']:
        assert [e['seed'] for e in case['episodes']]==report['evaluation_seeds']
        assert case['gait_passed']==sum(map(walking_pass,case['episodes']))
        for e in case['episodes']:
            assert len(e['gait']['steps_per_leg'])==4
            assert len(e['gait']['contact_fraction'])==4
    if report.get('refinement'):
        assert report['controllers']['varied']['training_steps']==report['refinement']['source_steps']+report['refinement']['config']['total_steps']
        assert (ROOT/'walking-study-initial.json').exists()


def test_archived_initial_weights_and_baseline_hashes_are_preserved():
    for name in ['walking-study.json','walking-study-initial.json']:
        r=json.loads((ROOT/name).read_text())
        for c in [*r['controllers'].values(),*r['baselines'].values()]:
            assert hashlib.sha256((ROOT/c['file']).read_bytes()).hexdigest()==c['sha256']

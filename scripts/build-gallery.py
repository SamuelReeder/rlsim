"""Package curated native policies and the complete fixed-budget experiment report."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from morpholab.portable import export_checkpoint
from morpholab.training import read_json, atomic_json

root=Path(__file__).resolve().parents[1]
output=root/'morpholab/static/demos';output.mkdir(parents=True,exist_ok=True)
entries=[]

def add(id,folder,title,description,checkpoint='latest',metrics=None):
    file=f'{id}.json'
    record=export_checkpoint(folder,output/file,checkpoint=checkpoint)
    entry={'id':id,'file':file,'title':title,'description':description+' These earlier tests measured survival and speed, not gait quality.','steps':record['steps'],
           'seed':500000007,'provenance':f"Native PPO, seed {record['config']['seed']}; portable browser inference",
           'source_run':folder.name,'checkpoint':checkpoint,
           'sha256':hashlib.sha256((output/file).read_bytes()).hexdigest()}
    if metrics:entry['native_test_metrics']=metrics
    entries.append(entry)

add('quadruped-walk',root/'runs/20261020T132800-fc7a3017','Quadruped · speed-only baseline',
    'A 524k-step policy. Change the target speed in Model to test its response.')
add('quadruped-balance',root/'runs/20261020T132516-493b1c07','Quadruped · early learning',
    'An independent 131k-step run that stays upright but makes little forward progress. Survival alone is not walking.')
report=read_json(root/'experiments/curriculum-pilot/report.json')
if not report.get('completed_at'):
    raise SystemExit('The complete experiment report is required before packaging the gallery.')
for body in ['biped','hexapod']:
    candidates=[t for t in report['trials'] if t['body']==body and t['status']=='completed' and
                t['metrics']['survival_rate']==1 and t['metrics']['distance']>5]
    if candidates:
        selected=min(candidates,key=lambda t:t['metrics']['speed_error'])
        add(body+'-walk',root/'experiments/curriculum-pilot/runs'/selected['run_id'],
            f'{body.title()} · speed-only baseline',
            f"Curated from the 65k-step pilot ({selected['condition'].lower()}). The complete comparison includes the unsuccessful trials too.",
            'best',selected['metrics'])
    else:
        extended=root/'experiments/gallery-hexapod/selected.json'
        if body=='hexapod' and extended.exists():
            selected=read_json(extended)
            folder=extended.parent/selected['run_id']
            state=read_json(folder/'status.json')
            if state.get('status')!='completed':
                raise SystemExit('Wait for the continued gallery run to finish.')
            add('hexapod-walk',folder,'Hexapod · continued training',
                'A 65k-step run continued for another 262k steps with its optimizer and normalization restored. This longer run is separate from the pilot comparison.',
                'best')
            entries[-1]['parent_run_id']=state['parent_run_id']
            entries[-1]['source_trial']=selected['source_trial']
            entries[-1]['training_steps']=state['steps']
        else:
            print(f'No {body} met the speed-and-survival gate.')
report['presentation_notes']=['Training ran on a shared development workstation; timings are not an isolated throughput benchmark.',
 'Domain randomization was enabled during training. Final tests use nominal physics with seeded initial states.',
 'Only the gallery is curated. This report retains every planned training trial.']
revisions=len({trial['source_hash'] for trial in report['trials']})
if revisions>1:
    report['presentation_notes'].append(f'This development pilot spans {revisions} training-code revisions; per-run hashes are included. Rerun at one fixed revision before making benchmark claims.')
atomic_json(output/'comparison.json',report)
atomic_json(output/'gallery.json',{'format':'rlsim-gallery-v1','entries':entries})
print(json.dumps({'demos':[entry['id'] for entry in entries],'trials':len(report['trials'])}))

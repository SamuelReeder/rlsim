"""Add the measured coordinated demos without erasing historical examples."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from morpholab.training import read_json,atomic_json
root=Path(__file__).resolve().parents[1]/'morpholab/static/demos'
r=read_json(root/'walking-study.json');g=read_json(root/'gallery.json')
entries=[]
for kind,title,description in [
 ('quadruped','Quadruped · coordinated walking','Learned position targets with alternating foot-contact rewards. Compare the steps by leg with the speed-only baseline.'),
 ('varied','Quadruped · coordinated proportions','All four legs receive stepping rewards across varied lengths. Use the proportion sliders above. The strict gait check also rejects excessive foot drift.')]:
 c=r['controllers'][kind]
 case=next(x for x in r['cases'] if x['backend']=='browser CPU' and x['controller']==kind and x['name']=='Nominal')
 episodes=case['episodes'];n=len(episodes)
 entries.append({'id':'coordinated-'+kind,'file':c['file'],'title':title,'description':description,
   'steps':c['steps'],'seed':940000007,'provenance':'Native PPO; '+str(c['training_steps'])+' total training steps; browser inference',
   'source_run':c['run_id'],'checkpoint':'best','sha256':c['sha256'],'thumbnail':'walking-'+kind+'.png',
   'browser_evaluation':{'device':'CPU','seeds':r['evaluation_seeds'],'episodes':episodes,
   'survived':sum(not e['fell'] for e in episodes),'mean_distance':sum(e['distance'] for e in episodes)/n,
   'mean_speed_error':sum(e['speed_error'] for e in episodes)/n,'gait_passed':case['gait_passed']}})
g['entries']=entries+[e for e in g['entries'] if e['id'] not in {'coordinated-quadruped','coordinated-varied'}]
atomic_json(root/'gallery.json',g)

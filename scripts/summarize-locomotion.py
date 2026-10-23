"""Summarize complete audits; efficiency comparisons use matched walking successes."""
import argparse,json,math,statistics
from pathlib import Path


def episodes(report):
    return {(tuple(c['scales']),c['speed'],e['seed']):e for c in report['cases'] for e in c['episodes']}


def summarize(report):
    rows=list(episodes(report).values());passed=[e for e in rows if e['gait_passed']]
    return {'passed':len(passed),'episodes':len(rows),'falls':sum(e['fell'] for e in rows),
      'mean_speed_error':statistics.mean(e['speed_error'] for e in rows),
      'median_drift_passing':statistics.median(e['gait']['foot_slip_m_s'] for e in passed) if passed else None,
      'median_work_passing_j':statistics.median(e['effort']['absolute_motor_work_j'] for e in passed) if passed and all('effort' in e for e in passed) else None}


def compare(baseline,candidate):
    a,b=episodes(baseline),episodes(candidate)
    if a.keys()!=b.keys():raise ValueError('Comparison must use identical shapes, commands and episode seeds.')
    common=[k for k in a if a[k]['gait_passed'] and b[k]['gait_passed']]
    integrated=all('effort' in a[k] and 'effort' in b[k] for k in common)
    group='effort' if integrated else 'physical'
    key='mechanical_cost_of_transport' if integrated else 'sampled_cost_of_transport'
    ratios=[b[k][group][key]/a[k][group][key] for k in common if a[k][group][key] and b[k][group][key] is not None]
    return {'baseline':summarize(baseline),'candidate':summarize(candidate),'matched_successes':len(common),
      'work_measurement':'physics-substep integrated' if integrated else '50 Hz endpoint estimate',
      'median_transport_cost_ratio_on_matched_successes':statistics.median(ratios) if ratios else None,
      'new_passes':sum(b[k]['gait_passed'] and not a[k]['gait_passed'] for k in a),
      'lost_passes':sum(a[k]['gait_passed'] and not b[k]['gait_passed'] for k in a)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();a,b=[json.loads(path.read_text()) for path in [args.baseline,args.candidate]]
    if not all(r.get('completed') or r.get('completed_at') for r in [a,b]):raise ValueError('Audit still incomplete.')
    result=compare(a,b);print(json.dumps(result,indent=2))
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()

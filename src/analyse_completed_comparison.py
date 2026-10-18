# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Audit a completed saved-prediction sweep and summarise its bounded evidence.

No inference, tuning or final-test access. Latency is a first-pass diagnostic.
"""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path

import numpy as np
from audit_dataset import digest

PROJECT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(PROJECT/'artifacts/matplotlib_config')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    args=parser.parse_args()
    if Path(args.run).name!=args.run or ':' in args.run or '\\' in args.run:
        raise ValueError('Unsafe run id')
    run=PROJECT/'artifacts'/args.run
    record_path=run/'run_record.json'
    record=json.loads(record_path.read_text(encoding='utf-8'))
    if record['status']!='completed' or record['purpose']!='development-study':
        raise RuntimeError('Require a completed development study')
    config_path=run/'source_snapshot/inference_comparison_development_v1.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if digest(config_path)!=record['config_sha256']:
        raise RuntimeError('Changed config snapshot')
    summary_path=run/'summary.json'
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    log=run/'predictions_and_timings.jsonl'
    if digest(log)!=record['prediction_log_sha256']:
        raise RuntimeError('Changed prediction log')
    expected_methods=[m['name'] for m in config['methods']]
    selected=record['selection']
    if len(selected)!=548 or len(set(selected))!=548 or set(summary)!=set(expected_methods):
        raise RuntimeError('Incomplete declared study')
    groups={name:[] for name in expected_methods}
    observed=set()
    for line in log.open(encoding='utf-8'):
        row=json.loads(line)
        key=(row['image'],row['method'])
        if key in observed or row['image'] not in selected or row['method'] not in groups:
            raise RuntimeError('Invalid image/method identity')
        observed.add(key)
        pred=np.asarray(row['predictions'],dtype=np.float64).reshape(-1,6)
        if not np.isfinite(pred).all() or np.any(pred[:,2:4]<=pred[:,:2]):
            raise RuntimeError('Nonfinite or nonpositive prediction geometry')
        if np.any((pred[:,4]<0)|(pred[:,4]>1)) or np.any(pred[:,5]!=np.floor(pred[:,5])) or np.any((pred[:,5]<0)|(pred[:,5]>9)):
            raise RuntimeError('Invalid confidence/class')
        if len(pred)>1 and np.any(np.diff(pred[:,4])>0):
            raise RuntimeError('Predictions not globally score sorted')
        t=row['timing']
        if any(not math.isfinite(t[k]) or t[k]<0 for k in ['decode_ms','views_ms','merge_ms','end_to_end_ms']):
            raise RuntimeError('Invalid timing')
        if not np.isclose(sum(t[k] for k in ['decode_ms','views_ms','merge_ms']),t['end_to_end_ms'],atol=1e-7,rtol=0):
            raise RuntimeError('Stage timings do not account for end-to-end total')
        counts=t['merge_counts']
        if not (len(pred)==t['prediction_count']==counts['after_cap'] and len(pred)<=config['max_detections']):
            raise RuntimeError('Prediction counts/cap inconsistent')
        if counts['before_merge']-counts['removed_nms']!=counts['after_nms_before_cap'] or counts['after_nms_before_cap']-counts['removed_cap']!=len(pred):
            raise RuntimeError('Merge accounting inconsistent')
        if t['detector_passes']!=len(t['per_view_counts']) or sum(x['output_after_geometry'] for x in t['per_view_counts'])!=counts['before_merge']:
            raise RuntimeError('View accounting inconsistent')
        groups[row['method']].append(t)
    if observed!={(name,method) for name in selected for method in expected_methods}:
        raise RuntimeError('Missing declared output')
    result={}
    for name in expected_methods:
        rows=groups[name]
        times=np.asarray([r['end_to_end_ms'] for r in rows])
        median,p95=float(np.median(times)),float(np.percentile(times,95))
        if not np.isclose(median,summary[name]['diagnostic_median_ms']) or not np.isclose(p95,summary[name]['diagnostic_p95_ms']):
            raise RuntimeError('Saved timing summary differs from raw records')
        metrics=summary[name]['scoring']['metrics']
        if summary[name]['scoring']['target_count']!=38759 or not all(math.isfinite(v) for v in metrics.values()):
            raise RuntimeError('Unexpected targets/nonfinite scoring summary')
        result[name]={'images':len(rows),'ap50_percent':100*metrics['metrics/mAP50(B)'],
                      'ap50_95_percent':100*metrics['metrics/mAP50-95(B)'],
                      'diagnostic_median_ms':median,'diagnostic_p95_ms':p95,
                      'mean_detector_passes':float(np.mean([r['detector_passes'] for r in rows])),
                      'total_detector_passes':sum(r['detector_passes'] for r in rows),
                      'total_processed_input_pixels':sum(r['processed_input_pixels'] for r in rows),
                      'output_cap_images':sum(r['prediction_count']==config['max_detections'] for r in rows),
                      'view_outputs_at_cap':sum(v['output_before_geometry']==config['max_detections'] for r in rows for v in r['per_view_counts']),
                      'boxes_removed_merge_nms':sum(r['merge_counts']['removed_nms'] for r in rows),
                      'boxes_removed_final_cap':sum(r['merge_counts']['removed_cap'] for r in rows),
                      'geometry_clipped':sum(r['geometry']['clipped'] for r in rows),
                      'geometry_zero_area_removed':sum(r['geometry']['zero_area_removed'] for r in rows),
                      'per_class_ap50_95_percent':{k:100*v for k,v in summary[name]['scoring']['per_class_ap'].items()}}
    baseline=result['full_640']
    for name,values in result.items():
        values['ap50_95_delta_vs_full640_pp']=values['ap50_95_percent']-baseline['ap50_95_percent']
        values['processed_input_pixel_ratio_vs_full640']=values['total_processed_input_pixels']/baseline['total_processed_input_pixels']
        values['diagnostic_median_time_ratio_vs_full640']=values['diagnostic_median_ms']/baseline['diagnostic_median_ms']
    dest=PROJECT/'artifacts'/(args.run+'_analysis')
    dest.mkdir(exist_ok=False)
    output={'utc':datetime.now(timezone.utc).isoformat(),'input_run':args.run,'image_method_records_checked':len(observed),
            'input_hashes':{'record':digest(record_path),'summary':digest(summary_path),'predictions':digest(log),'config':digest(config_path)},
            'analysis_source_sha256':digest(Path(__file__)),'methods':result,
            'limits':['Exposed validation, one selected training seed, exploratory operating points',
                      'Shared 640-trained weights; not best training at every resolution',
                      'Single-label prediction policy; not training-validator metrics',
                      'One pass per image on desktop; timing ratios are preliminary, not stable speedup claims',
                      'Aggregate AP does not establish small-object or field-task gains',
                      'Converted protocol; not native VisDrone score or final test evaluation']}
    (dest/'summary.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    (dest/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
    labels={'full_640':'Full 640','full_960':'Full 960','full_1280':'Full 1280','fixed_slices_640_plus_global':'Fixed slices + global'}
    for name,v in result.items():
        for axis,x in [(axes[0],v['processed_input_pixel_ratio_vs_full640']),(axes[1],v['diagnostic_median_ms'])]:
            axis.scatter(x,v['ap50_95_percent'],s=42)
            axis.annotate(labels[name],(x,v['ap50_95_percent']),xytext=(5,5),textcoords='offset points',fontsize=9)
    axes[0].set_xlabel('Processed square-input pixels relative to full 640')
    axes[1].set_xlabel('First-pass median end-to-end time (ms)')
    for axis in axes:
        axis.set_ylabel('Converted development AP50-95 (%)')
        axis.grid(alpha=.2)
        axis.margins(x=.25,y=.18)
    fig.suptitle('One checkpoint / 548 development images | exploratory, not onboard or final-test evidence',fontsize=10)
    fig.savefig(dest/'operating_points.png',dpi=180)
    plt.close(fig)
    print(json.dumps({'records_checked':len(observed),'methods':result},indent=2))


if __name__=='__main__':
    main()

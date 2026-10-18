# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Audit three same-checkpoint sweeps and summarise repeated desktop timing.

AP repetition is not training replication. No best-run filtering is performed.
"""
import json
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from audit_dataset import digest

PROJECT=Path(__file__).resolve().parents[1]
RUNS=['d001_uvd1_sharedweights_inference_20261016','d001_timing_repeat2_20261016','d001_timing_repeat3_20261016']


def main():
    records=[]
    summaries=[]
    timings=[]
    predictions={}
    equality=[]
    fingerprints=[]
    for run_index,name in enumerate(RUNS):
        run=PROJECT/'artifacts'/name
        record=json.loads((run/'run_record.json').read_text(encoding='utf-8'))
        if record['status']!='completed' or record['purpose']!='development-study' or record['images_compared']!=548:
            raise RuntimeError('All three complete development sweeps are required')
        if run_index:
            for key in ['weights_sha256','config_sha256','dataset_manifest_sha256','selection','predict_arguments','source_hashes','framework_source_hashes']:
                if record[key]!=records[0][key]:
                    raise RuntimeError('Repeat input/protocol differs: '+key)
        log=run/'predictions_and_timings.jsonl'
        if digest(log)!=record['prediction_log_sha256']:
            raise RuntimeError('Saved prediction log changed')
        records.append(record)
        summaries.append(json.loads((run/'summary.json').read_text(encoding='utf-8')))
        fingerprint={'run':name,'record_sha256':digest(run/'run_record.json'),'summary_sha256':digest(run/'summary.json'),'predictions_sha256':digest(log)}
        fingerprints.append(fingerprint)
        times={}
        comparison={'run':name,'records_compared':0,'arrays_differing':0,'shape_differences':0,'max_abs_same_shape_coordinate_difference':0.,'max_abs_same_shape_confidence_difference':0.}
        for line in log.open(encoding='utf-8'):
            row=json.loads(line)
            key=(row['image'],row['method'])
            if key in times:
                raise RuntimeError('Duplicate image/method')
            elapsed=row['timing']['end_to_end_ms']
            if not np.isfinite(elapsed) or elapsed<0:
                raise RuntimeError('Invalid elapsed time')
            times[key]=elapsed
            value=np.asarray(row['predictions'],dtype=np.float32).reshape(-1,6)
            if run_index==0:
                predictions[key]=value
            else:
                if key not in predictions:
                    raise RuntimeError('Unexpected prediction identity')
                comparison['records_compared']+=1
                ref=predictions[key]
                if not np.array_equal(value,ref):
                    comparison['arrays_differing']+=1
                    if value.shape!=ref.shape:
                        comparison['shape_differences']+=1
                    elif len(value):
                        comparison['max_abs_same_shape_coordinate_difference']=max(comparison['max_abs_same_shape_coordinate_difference'],float(np.abs(value[:,:4]-ref[:,:4]).max()))
                        comparison['max_abs_same_shape_confidence_difference']=max(comparison['max_abs_same_shape_confidence_difference'],float(np.abs(value[:,4]-ref[:,4]).max()))
        expected={(image,method) for image in record['selection'] for method in summaries[0]}
        if set(times)!=expected:
            raise RuntimeError('Incomplete timing outputs')
        timings.append(times)
        if run_index:equality.append(comparison)
    result={}
    for method in summaries[0]:
        matrix=np.asarray([[t[(image,method)] for image in records[0]['selection']] for t in timings])
        medians=np.median(matrix,axis=1)
        per_image=np.median(matrix,axis=0)
        ap=[100*s[method]['scoring']['metrics']['metrics/mAP50-95(B)'] for s in summaries]
        result[method]={'image_count':548,'sweeps':3,'per_sweep_median_ms':medians.tolist(),
                        'per_sweep_median_range_ms':[float(medians.min()),float(medians.max())],
                        'median_of_per_image_three_sweep_medians_ms':float(np.median(per_image)),
                        'p95_of_per_image_three_sweep_medians_ms':float(np.percentile(per_image,95)),
                        'ap50_95_percent_per_sweep':ap,'ap50_95_range_pp':max(ap)-min(ap)}
    output={'utc':datetime.now(timezone.utc).isoformat(),'runs':fingerprints,'methods':result,'prediction_comparisons_to_first':equality,
            'analysis_source_sha256':digest(Path(__file__)),
            'limits':['Same trained checkpoint and exposed validation images, not independent training seeds',
                      'Within-image three-sweep median then cross-image median/p95; no best-run filtering',
                      'Same Windows desktop; no intentional concurrent training/native scoring; other desktop load and clocks uncontrolled',
                      'Cached-filesystem end-to-end boundary as declared; not onboard or universal timing evidence']}
    dest=PROJECT/'artifacts/d001_timing_repeats_analysis_20261016'
    dest.mkdir(exist_ok=False)
    (dest/'summary.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    (dest/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()

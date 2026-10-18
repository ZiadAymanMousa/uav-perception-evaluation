# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
# Matching pair extraction follows Ultralytics contributors' AGPL-3.0 implementation; see THIRD_PARTY_NOTICES.md.
"""Exploratory full-population GT recall by original size from immutable saved detections."""
import csv
from datetime import datetime, timezone
import inspect
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
from audit_dataset import digest
from inference_geometry import prediction_rows
from prepare_training_pilot import convert_rows, NAMES
from score_converted_predictions import label_rows_as_xyxy
from study_inputs import load_cases

PROJECT = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(PROJECT/'artifacts/ultralytics_config')
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['MPLCONFIGDIR'] = str(PROJECT/'artifacts/matplotlib_config')
POINTS = [('c025_i050', .25, .5), ('saved_i050', None, .5), ('c025_i075', .25, .75)]
BINS = ['a_lt_256','a_256_to_1024','a_1024_to_9216','a_ge_9216']


def area_bin(area):
    if np.any(np.asarray(area) <= 0):
        raise ValueError('Expected positive original area')
    return np.searchsorted([256,1024,9216], area, side='right')


def pairs(masked_iou, threshold):
    """Expose pairs using the inspected pinned non-SciPy order, including unique ordering."""
    matches = np.array(np.nonzero(masked_iou >= threshold)).T
    if len(matches) > 1:
        matches = matches[masked_iou[matches[:,0],matches[:,1]].argsort()[::-1]]
        matches = matches[np.unique(matches[:,1],return_index=True)[1]]
        matches = matches[np.unique(matches[:,0],return_index=True)[1]]
    return matches.astype(np.int64)


def controls():
    assert area_bin(np.array([1,255,256,1023,1024,9215,9216])).tolist() == [0,0,1,1,2,2,3]
    # Both target sizes remain present when reporting a small stratum.
    p = pairs(np.array([[1.,0.],[0.,1.]],dtype=np.float32),.5)
    assert p.tolist() == [[0,0],[1,1]]
    # Duplicate detections do not produce two target hits; no candidate is not a hit.
    assert pairs(np.array([[1.,.9]],dtype=np.float32),.5).tolist() == [[0,0]]
    assert pairs(np.zeros((2,0),dtype=np.float32),.5).shape == (0,2)
    assert pairs(np.zeros((1,1),dtype=np.float32),.5).shape == (0,2)
    # A crowded assignment is deliberately not assumed to be maximum cardinality.
    crowded = np.array([[.8,.7],[.9,0]],dtype=np.float32)
    assert pairs(crowded,.5).tolist() == [[0,1],[1,0]]
    # The pinned unique order takes pred0 here despite pred1's larger IoU.
    assert pairs(np.array([[.8,1.]],dtype=np.float32),.5).tolist() == [[0,0]]
    return {'area_boundaries':True,'all_size_population':True,'duplicate_empty_wrong_class':True,
            'crowded_pairs':True,'pinned_unique_order':True}


def main():
    import torch
    import ultralytics
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import box_iou
    assert ultralytics.__version__ == '8.3.253'
    torch.set_num_threads(1)
    dest = PROJECT/'artifacts/e001_d001_size_errors_20261017'
    dest.mkdir(exist_ok=False)
    source = Path(__file__)
    shutil.copyfile(source,dest/'analysis_source.py')
    plan = PROJECT/'Size_Error_Analysis_Plan_2026-10-17.md'
    shutil.copyfile(plan,dest/plan.name)
    record = {'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
              'analysis_source_sha256':digest(source),'plan_sha256':digest(plan)}
    def save_record():
        (dest/'run_record.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    save_record()
    started = time.perf_counter()
    try:
        check = controls()
        methods, provenance = load_cases('d001_uvd1_sharedweights_inference_20261016')
        names = list(methods)
        by_image = {name:{x['name']+'.jpg':x for x in cases} for name,cases in methods.items()}
        manifest = json.loads((PROJECT/'data/yolo_full_uvd1/manifest.json').read_text())
        entries = {x['image']:x for x in manifest['splits']['val']['entries']}
        comparisons = list(combinations(names,2))
        denominators = np.zeros((4,10),dtype=np.int64)
        hits = {(point,name):np.zeros((4,10),dtype=np.int64) for point,_,_ in POINTS for name in names}
        global_counts = {(point,name):{'eligible':0,'tp':0,'fp':0} for point,_,_ in POINTS for name in names}
        paired = {(point,a,b):np.zeros((4,4),dtype=np.int64) for point,_,_ in POINTS for a,b in comparisons}
        identities = {point:0 for point,_,_ in POINTS}
        matcher = lambda threshold:SimpleNamespace(iouv=torch.tensor([threshold],dtype=torch.float32))
        record.update(source=provenance,operating_points=POINTS,area_edges_px2=[0,256,1024,9216,None],
                      method_order=names, controls=check,software={'torch':torch.__version__,'ultralytics':ultralytics.__version__},
                      matcher_source_sha256=digest(Path(inspect.getfile(BaseValidator))),
                      iou_source_sha256=digest(Path(inspect.getfile(box_iou))))
        save_record()
        with (dest/'gt_catalog.jsonl').open('w',encoding='utf-8') as catalogue, (dest/'matches_by_image.jsonl').open('w',encoding='utf-8') as output:
            for index,(image,entry) in enumerate(entries.items(),1):
                first = by_image[names[0]][image]
                width,height = first['width'],first['height']
                annotation = PROJECT/'data/VisDrone2019-DET-val/annotations'/f'{Path(image).stem}.txt'
                label = PROJECT/'data/yolo_full_uvd1/labels/val'/f'{Path(image).stem}.txt'
                assert digest(annotation) == entry['source_annotation_sha256']
                assert digest(label) == entry['label_sha256']
                source_rows = [(number,next(csv.reader([line]))) for number,line in enumerate(annotation.read_text().splitlines(),1) if line.strip()]
                converted,excluded,clipped = convert_rows([row for _,row in source_rows],width,height)
                assert not clipped
                serialised = ''.join(f'{row[0]} '+' '.join(f'{v:.9f}' for v in row[1:])+'\n' for row in converted)
                assert serialised == label.read_text(), 'Source-to-converted mapping changed'
                raw_targets = [(number,np.asarray(row,dtype=np.int64)) for number,row in source_rows if float(row[4]) == 1 and 1 <= int(row[5]) <= 10]
                gt = label_rows_as_xyxy(label,width,height)
                assert len(gt) == len(raw_targets) == entry['targets']
                cls = gt[:,0].astype(np.int64)
                areas = np.array([row[2]*row[3] for _,row in raw_targets],dtype=np.int64)
                bins = area_bin(areas)
                np.add.at(denominators,(bins,cls),1)
                for j,(number,row) in enumerate(raw_targets):
                    catalogue.write(json.dumps({'image':image,'gt_id':f'{image}:{number}','source_line_1based':number,
                        'converted_row_0based':j,'class0':int(cls[j]),'source_category':int(row[5]),'raw_xywh':row[:4].tolist(),
                        'matching_xyxy':gt[j,1:].tolist(),'area_original_px2':int(areas[j]),'area_bin':BINS[int(bins[j])],
                        'image_width':width,'image_height':height,'truncation':int(row[6]),'occlusion':int(row[7])})+'\n')
                states = {}
                for name in names:
                    # Native bridge stores xywh/conf/class1..10; reconstruct exact saved float32 rows.
                    native = np.asarray(by_image[name][image]['detections'],dtype=np.float64).reshape(-1,8)
                    pred = np.column_stack((native[:,:2],native[:,:2]+native[:,2:4],native[:,4],native[:,5]-1)).astype(np.float32)
                    pred = prediction_rows(pred)
                    overlap = box_iou(torch.from_numpy(gt[:,1:]),torch.from_numpy(pred[:,:4]))
                    for point,confidence,threshold in POINTS:
                        eligible = np.arange(len(pred)) if confidence is None else np.flatnonzero(pred[:,4]>=confidence)
                        chosen = pred[eligible]
                        ov = overlap[:,eligible]
                        same = cls[:,None] == chosen[:,5]
                        masked = ov.numpy()*same
                        matched = pairs(masked,threshold)
                        flags = np.zeros(len(chosen),dtype=bool)
                        flags[matched[:,1]] = True
                        actual = BaseValidator.match_predictions(matcher(threshold),torch.from_numpy(chosen[:,5]),torch.from_numpy(gt[:,0]),ov).numpy()[:,0]
                        assert np.array_equal(flags,actual), 'Pinned TP parity failure'
                        target_hit = np.zeros(len(gt),dtype=bool)
                        target_hit[matched[:,0]] = True
                        assert len(np.unique(matched[:,0])) == len(matched) == len(np.unique(matched[:,1]))
                        assert np.all(masked[matched[:,0],matched[:,1]]>=threshold)
                        np.add.at(hits[(point,name)],(bins[target_hit],cls[target_hit]),1)
                        totals = global_counts[(point,name)]
                        totals['eligible'] += len(chosen); totals['tp'] += len(matched); totals['fp'] += len(chosen)-len(matched)
                        states[(point,name)] = target_hit
                        identities[point] += 1
                        output.write(json.dumps({'image':image,'method':name,'operating_point':point,
                            'eligible_saved_ranks_0based':eligible.tolist(),'matched_gt_rows_0based':matched[:,0].tolist(),
                            'matched_saved_ranks_0based':eligible[matched[:,1]].tolist(),
                            'matched_iou':masked[matched[:,0],matched[:,1]].tolist(),
                            'matched_confidence':chosen[matched[:,1],4].tolist(),'n_gt':len(gt),
                            'n_tp':len(matched),'n_fp':len(chosen)-len(matched),'saved_output_at_cap':len(pred)==1000})+'\n')
                for point,_,_ in POINTS:
                    for a,b in comparisons:
                        A,B = states[(point,a)],states[(point,b)]
                        for k,mask in enumerate([A&B,A&~B,~A&B,~A&~B]):
                            paired[(point,a,b)][:,k] += np.bincount(bins[mask],minlength=4)
                if index % 50 == 0:
                    print('SIZE_PROGRESS '+str(index)+'/548',flush=True)
        assert denominators.sum() == 38759
        assert all(value == 548*4 for value in identities.values())
        result = {'scope':'One selected development checkpoint; converted-task original-size GT recall, not size AP',
                  'denominators_by_bin_class':denominators.tolist(),'class_names':NAMES,'bin_ids':BINS,'methods':{},'paired':{}}
        for point,_,_ in POINTS:
            result['methods'][point] = {}
            result['paired'][point] = {}
            for name in names:
                matched = hits[(point,name)]
                total = global_counts[(point,name)]
                assert matched.sum() == total['tp']
                micro = np.divide(matched.sum(1),denominators.sum(1),out=np.full(4,np.nan),where=denominators.sum(1)>0)
                macro = [float(np.mean(matched[i,denominators[i]>0]/denominators[i,denominators[i]>0])) for i in range(4)]
                result['methods'][point][name] = {'matched_by_bin_class':matched.tolist(),'micro_recall_by_bin':micro.tolist(),
                    'equal_class_mean_recall_by_nonempty_bin':macro,'global':dict(total,precision=total['tp']/max(total['eligible'],1),recall=total['tp']/38759)}
            for a,b in comparisons:
                values = paired[(point,a,b)]
                assert np.array_equal(values.sum(1),denominators.sum(1))
                assert np.array_equal(values[:,2]-values[:,1],hits[(point,b)].sum(1)-hits[(point,a)].sum(1))
                result['paired'][point][a+'__vs__'+b] = {'columns':['both','A_only','B_only','neither'],'counts_by_bin':values.tolist()}
        (dest/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        record.update(status='completed',target_count=int(denominators.sum()),images=548,methods=4,operating_points_count=3,
                      exact_framework_tp_checks=sum(identities.values()),
                      output_hashes={p.name:digest(p) for p in dest.iterdir() if p.suffix in ('.jsonl',) or p.name=='summary.json'},
                      limits=['exploratory exposed development data','converted ignore policy','saved capped post-NMS outputs only','no causal miss attribution or selector utility','no independent scene or cross-seed claim','no small-object AP'])
    except BaseException as exc:
        record.update(status='failed',error=repr(exc))
        raise
    finally:
        record.update(seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        save_record()
    print('SIZE_COMPLETE '+json.dumps({'targets':record['target_count'],'tp_checks':record['exact_framework_tp_checks']}),flush=True)


if __name__ == '__main__':
    main()

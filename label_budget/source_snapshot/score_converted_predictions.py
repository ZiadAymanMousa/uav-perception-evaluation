# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Pinned Ultralytics matching/AP on saved original-image predictions.

Ten-class converted protocol only: no native ignore semantics or official COCO
equivalence claim. Input prediction rows are xyxy/confidence/class 0..9.
"""
from types import SimpleNamespace

import numpy as np

from inference_geometry import prediction_rows


def label_rows_as_xyxy(path,width,height):
    if not path.read_text(encoding='utf-8').strip():
        return np.empty((0,5),dtype=np.float32)
    values=np.loadtxt(path,dtype=np.float32,ndmin=2)
    if values.shape[1]!=5 or not np.isfinite(values).all():
        raise ValueError('Expected finite YOLO labels')
    if np.any(values[:,0]!=np.floor(values[:,0])) or np.any((values[:,0]<0)|(values[:,0]>9)):
        raise ValueError('Invalid target classes')
    if np.any(values[:,3:5]<=0) or np.any((values[:,1:]<0)|(values[:,1:]>1)):
        raise ValueError('Invalid normalised geometry')
    output=values.copy()
    cx,cy=values[:,1]*width,values[:,2]*height
    bw,bh=values[:,3]*width,values[:,4]*height
    output[:,1]=cx-bw/2
    output[:,2]=cy-bh/2
    output[:,3]=cx+bw/2
    output[:,4]=cy+bh/2
    return output


def score_records(records,names):
    """Reuse installed pinned matching/AP functions; records are GT + predictions."""
    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import DetMetrics,box_iou
    metrics=DetMetrics(names=names)
    matcher=SimpleNamespace(iouv=torch.linspace(.5,.95,10))
    target_count=0
    for record in records:
        gt=np.asarray(record['gt'],dtype=np.float32).reshape(-1,5)
        pred=prediction_rows(record['predictions'])
        if not np.isfinite(gt).all() or np.any(gt[:,3:5]<=gt[:,1:3]):
            raise ValueError('Invalid GT xyxy')
        if np.any(gt[:,0]!=np.floor(gt[:,0])) or np.any((gt[:,0]<0)|(gt[:,0]>9)):
            raise ValueError('Invalid GT class')
        if len(pred)>1 and np.any(np.diff(pred[:,4])>0):
            raise ValueError('Predictions must be globally score-sorted')
        target_count+=len(gt)
        tp=np.zeros((len(pred),10),dtype=bool)
        if len(gt) and len(pred):
            overlap=box_iou(torch.from_numpy(gt[:,1:5]),torch.from_numpy(pred[:,:4]))
            tp=BaseValidator.match_predictions(matcher,torch.from_numpy(pred[:,5]),torch.from_numpy(gt[:,0]),overlap).numpy()
        metrics.update_stats({'tp':tp,'conf':pred[:,4],'pred_cls':pred[:,5],
                              'target_cls':gt[:,0],'target_img':np.unique(gt[:,0])})
    if target_count==0:
        return {'status':'no targets; AP undefined','target_count':0,'metrics':None}
    metrics.process(plot=False)
    return {'status':'converted-protocol diagnostic','target_count':target_count,
            'metrics':{k:float(v) for k,v in metrics.results_dict.items()},
            'per_class_ap':{names[int(index)]:float(value) for index,value in zip(metrics.ap_class_index,metrics.box.ap)},
            'protocol':'Ultralytics matching/AP in original image coordinates; not native VisDrone or official COCO'}

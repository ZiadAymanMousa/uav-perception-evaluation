# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Finite-pool label-budget selector. Receives revealed sufficient statistics only.

Private exploratory diagnostic; no new estimator or population guarantee.
"""
from dataclasses import dataclass
from typing import Mapping
import numpy as np

METHODS=('full_640','full_960','full_1280','fixed_slices_640_plus_global')
NAMES=('pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor')

@dataclass(frozen=True)
class Stats:
    tp: np.ndarray
    conf: np.ndarray
    pred_cls: np.ndarray
    target_cls: np.ndarray

@dataclass(frozen=True)
class Revealed:
    image_ids: tuple
    stats: Mapping
    target_classes: tuple
    target_count: int


def select_scores(scores, order=METHODS):
    """Exact floating-point ties follow the prespecified method order; no epsilon."""
    if tuple(scores)!=tuple(order):
        raise ValueError('Require precisely the declared ordered candidates')
    if not all(np.isfinite(v) for v in scores.values()):
        raise ValueError('Nonfinite candidate score')
    return max(order,key=lambda method:scores[method])


def _aggregate(rows):
    if not rows:
        raise ValueError('No queried images')
    return {'tp':np.concatenate([r.tp for r in rows]),
            'conf':np.concatenate([r.conf for r in rows]),
            'pred_cls':np.concatenate([r.pred_cls for r in rows]),
            'target_cls':np.concatenate([r.target_cls for r in rows])}


def score_revealed(revealed, order=METHODS):
    """Only a labelled subset enters. Incomplete class coverage => abstain.

    Pooled AP remains a potentially biased plug-in subset score. It is not the
    mean of per-image AP, an unbiased estimator, or a confidence guarantee.
    """
    if tuple(revealed.stats)!=tuple(order):
        raise ValueError('Candidate identities/order differ')
    if len(set(revealed.image_ids))!=len(revealed.image_ids) or not revealed.image_ids:
        raise ValueError('Require nonempty unique queried image identities')
    if any(len(rows)!=len(revealed.image_ids) for rows in revealed.stats.values()):
        raise ValueError('Incomplete queried candidate/image statistics')
    observed=tuple(sorted(set(np.concatenate([r.target_cls for r in revealed.stats[order[0]]]).astype(int).tolist())))
    if observed!=revealed.target_classes:
        raise ValueError('Coverage receipt differs from queried target classes')
    if revealed.target_count!=sum(len(r.target_cls) for r in revealed.stats[order[0]]):
        raise ValueError('Target count receipt differs from queried labels')
    reference=[r.target_cls for r in revealed.stats[order[0]]]
    for method in order:
        for r,gt in zip(revealed.stats[method],reference):
            if not np.array_equal(r.target_cls,gt):
                raise ValueError('Candidates must receive identical queried GT')
            if r.tp.shape!=(len(r.conf),10) or r.pred_cls.shape!=r.conf.shape:
                raise ValueError('Malformed sufficient statistics')
            if r.tp.dtype!=np.bool_ or not np.isfinite(r.conf).all():
                raise ValueError('Invalid TP/confidence statistics')
    missing=tuple(i for i in range(10) if i not in observed)
    base={'queried_images':len(revealed.image_ids),'target_count':revealed.target_count,
          'represented_classes':list(observed),'missing_classes':list(missing)}
    if missing:
        return dict(base,status='abstained_missing_class',selected=None,scores_ap50_95_percent=None)
    # Import only the pinned CPU AP implementation; no oracle/pool/file access.
    from ultralytics.utils.metrics import ap_per_class
    scores={}
    for method in order:
        stats=_aggregate(revealed.stats[method])
        returned=ap_per_class(**stats,plot=False,names=dict(enumerate(NAMES)))
        class_ids=tuple(returned[6].astype(int).tolist())
        if class_ids!=tuple(range(10)):
            raise RuntimeError('AP denominator is not the fixed ten-class task')
        ap=returned[5]
        if ap.shape!=(10,10) or not np.isfinite(ap).all():
            raise RuntimeError('Unexpected AP shape or nonfinite metric')
        scores[method]=float(100*ap.mean())
    return dict(base,status='selected',selected=select_scores(scores,order),scores_ap50_95_percent=scores)


class CacheOracle:
    """Infrastructure boundary, not a security sandbox. Only reveal() feeds selection.

    Image indices are sorted to source order before concatenation: random query
    order must not change AP through equal-score sorting ties.
    """
    def __init__(self,image_ids,by_method):
        self._image_ids=tuple(image_ids)
        self._by_method=by_method
        if len(set(image_ids))!=len(image_ids):
            raise ValueError('Duplicate pool identities')
        if tuple(by_method)!=METHODS or any(len(rows)!=len(image_ids) for rows in by_method.values()):
            raise ValueError('Incomplete pool/candidate coverage')

    def reveal(self,indices):
        indices=tuple(indices)
        if not indices or any(type(i) is not int or i<0 or i>=len(self._image_ids) for i in indices):
            raise ValueError('Invalid query indices')
        if len(set(indices))!=len(indices):
            raise ValueError('Repeated query image')
        ordered=tuple(sorted(indices))
        rows={method:tuple(self._by_method[method][i] for i in ordered) for method in METHODS}
        first=rows[METHODS[0]]
        classes=tuple(sorted(set(np.concatenate([r.target_cls for r in first]).astype(int).tolist())))
        return Revealed(tuple(self._image_ids[i] for i in ordered),rows,classes,sum(len(r.target_cls) for r in first))

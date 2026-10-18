# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Geometry and class-aware merging for a controlled fixed-slicing comparison."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Window:
    x1:int
    y1:int
    x2:int
    y2:int


def fixed_windows(width,height,tile_size=640,overlap=.2):
    if min(width,height,tile_size)<=0 or not all(isinstance(v,int) for v in (width,height,tile_size)):
        raise ValueError('Positive integer image/tile dimensions required')
    if not math.isfinite(overlap) or not 0<=overlap<1:
        raise ValueError('Overlap must be finite in [0,1)')
    step=max(1,int(tile_size*(1-overlap)))

    def starts(extent):
        last=max(0,extent-tile_size)
        values=list(range(0,last+1,step))
        if values[-1]!=last:
            values.append(last)
        return values

    return [Window(x,y,min(width,x+tile_size),min(height,y+tile_size))
            for y in starts(height) for x in starts(width)]


def prediction_rows(value):
    """Validate xyxy, confidence, zero-based VisDrone class; preserve float32."""
    rows=np.asarray(value,dtype=np.float32)
    if rows.size==0:
        return np.empty((0,6),dtype=np.float32)
    if rows.ndim!=2 or rows.shape[1]!=6 or not np.isfinite(rows).all():
        raise ValueError('Expected finite Nx6 predictions')
    if np.any(rows[:,2:4]<rows[:,:2]):
        raise ValueError('Reversed prediction box')
    if np.any((rows[:,4]<0)|(rows[:,4]>1)):
        raise ValueError('Confidence outside [0,1]')
    if np.any(rows[:,5]!=np.floor(rows[:,5])) or np.any((rows[:,5]<0)|(rows[:,5]>9)):
        raise ValueError('Expected zero-based ten-class predictions')
    return rows.copy()


def to_image_coordinates(value,window):
    rows=prediction_rows(value)
    if window.x1<0 or window.y1<0 or window.x2<=window.x1 or window.y2<=window.y1:
        raise ValueError('Invalid crop window')
    original=rows[:,:4].copy()
    rows[:,[0,2]]=np.clip(rows[:,[0,2]],0,window.x2-window.x1)
    rows[:,[1,3]]=np.clip(rows[:,[1,3]],0,window.y2-window.y1)
    clipped=int(np.any(original!=rows[:,:4],axis=1).sum())
    valid=(rows[:,2]>rows[:,0])&(rows[:,3]>rows[:,1])
    removed=int((~valid).sum())
    rows=rows[valid]
    rows[:,:4]+=np.asarray([window.x1,window.y1,window.x1,window.y1],dtype=np.float32)
    return rows,{'clipped':clipped,'zero_area_removed':removed}


def merge_predictions(parts,iou=.7,max_det=1000,*,return_counts=False):
    """CPU class-aware NMS and global score cap, with merge cost timed by caller."""
    if not 0<iou<=1 or not isinstance(max_det,int) or max_det<1:
        raise ValueError('Invalid NMS/cap')
    rows=np.concatenate([prediction_rows(x) for x in parts],axis=0) if parts else np.empty((0,6),dtype=np.float32)
    valid=(rows[:,2]>rows[:,0])&(rows[:,3]>rows[:,1])
    if not valid.all():
        raise ValueError('Zero-area boxes must be handled before merging')
    if len(rows)==0:
        counts={key:0 for key in ('before_merge','after_nms_before_cap','after_cap','removed_nms','removed_cap')}
        return (rows,counts) if return_counts else rows
    import torch
    from torchvision.ops import batched_nms
    tensor=torch.from_numpy(rows)
    keep=batched_nms(tensor[:,:4],tensor[:,4],tensor[:,5].to(torch.int64),iou)
    result=rows[keep[:max_det].numpy()]
    counts={'before_merge':len(rows),'after_nms_before_cap':len(keep),'after_cap':len(result),
            'removed_nms':len(rows)-len(keep),'removed_cap':len(keep)-len(result)}
    return (result,counts) if return_counts else result

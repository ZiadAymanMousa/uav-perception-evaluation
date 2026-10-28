# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Fixed descriptive CPU geometry diagnostic; established SIFT/RANSAC, no classifier."""
from __future__ import annotations
import math
import numpy as np
import cv2



def configure_cpu(seed=20261027):
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    cv2.setRNGSeed(seed)
    if cv2.ocl.useOpenCL():
        raise RuntimeError('OpenCL must remain disabled')


def resize_rgb(rgb, max_edge=1280):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected uint8 RGB')
    h,w=rgb.shape[:2]
    scale=min(1.,max_edge/max(w,h))
    nw=max(1,int(round(w*scale))); nh=max(1,int(round(h*scale)))
    # Pixel-centre mapping, using actual rounded dimension scale factors.
    sx=nw/w;sy=nh/h
    T=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]],dtype=np.float64)
    out=rgb.copy() if (nw,nh)==(w,h) else cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA)
    return out,T


def project(points,H):
    points=np.asarray(points,dtype=np.float64).reshape(-1,2)
    homogeneous=np.c_[points,np.ones(len(points))]@np.asarray(H,dtype=np.float64).T
    valid=np.isfinite(homogeneous).all(axis=1)&(np.abs(homogeneous[:,2])>1e-12)
    result=np.full((len(points),2),np.nan)
    result[valid]=homogeneous[valid,:2]/homogeneous[valid,2,None]
    return result


def safe(v):
    return float(v) if np.isfinite(v) else None


def inverse_matrix(H,condition_max=1e12):
    if H is None or np.shape(H)!=(3,3) or not np.isfinite(H).all():
        return None,None,'absent_or_nonfinite'
    condition=float(np.linalg.cond(H))
    if not np.isfinite(condition) or condition>condition_max:
        return None,safe(condition),'singular_or_ill_conditioned'
    try:
        inverse=np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None,safe(condition),'singular'
    return inverse,condition,'invertible'


def create_transform(rgb,kind):
    h,w=rgb.shape[:2]
    if min(h,w)<12:raise ValueError('Control source too small')
    if kind=='scale_crop':
        x0=int(math.floor(.1*w));y0=int(math.floor(.1*h))
        x1=w-x0;y1=h-y0
        sx=w/(x1-x0);sy=h/(y1-y0)
        H=np.array([[sx,0,sx*(.5-x0)-.5],[0,sy,sy*(.5-y0)-.5],[0,0,1.]],dtype=np.float64)
    elif kind=='rotation':
        H=np.vstack([cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),8.,1.),[0,0,1.]])
    elif kind=='perspective':
        src=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        dst=np.float32([[.04*w,.03*h],[.94*w,.02*h],[.97*w,.94*h],[.03*w,.97*h]])
        H=cv2.getPerspectiveTransform(src,dst)
    else:raise ValueError('Unknown fixed control transform')
    image=cv2.warpPerspective(rgb,H,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    mask=cv2.warpPerspective(np.full((h,w),255,np.uint8),H,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return image,H,mask


def extract_features(rgb):
    resized,T=resize_rgb(rgb)
    grey=cv2.cvtColor(resized,cv2.COLOR_RGB2GRAY)
    detector=cv2.SIFT_create(nfeatures=4000,nOctaveLayers=3,contrastThreshold=.04,edgeThreshold=10.,sigma=1.6)
    keypoints,descriptors=detector.detectAndCompute(grey,None)
    if descriptors is None:descriptors=np.empty((0,128),np.float32)
    points=np.array([k.pt for k in keypoints],np.float64).reshape(-1,2)
    original=project(points,np.linalg.inv(T))
    records=[]
    for i,k in enumerate(keypoints):
        records.append({'index':i,'x_resized':float(k.pt[0]),'y_resized':float(k.pt[1]),'x_original':float(original[i,0]),'y_original':float(original[i,1]),'size_resized':float(k.size),'angle_degrees':float(k.angle),'response':float(k.response),'octave':int(k.octave),'class_id':int(k.class_id)})
    return {'resized_rgb':resized,'T_original_to_resized':T,'original_shape_hw':list(rgb.shape[:2]),'resized_shape_hw':list(resized.shape[:2]),'keypoints':records,'points_resized':points,'points_original':original,'descriptors':descriptors}


def knn_rows(query_descriptors,train_descriptors,direction):
    # A query without2 valid neighbours is retained but cannot pass the ratio.
    if len(query_descriptors)==0:return []
    if len(train_descriptors)==0:matches=[]
    else:matches=cv2.BFMatcher(cv2.NORM_L2,crossCheck=False).knnMatch(query_descriptors,train_descriptors,k=min(2,len(train_descriptors)))
    by_query={items[0].queryIdx:items for items in matches if items}
    result=[]
    for qi in range(len(query_descriptors)):
        items=by_query.get(qi,[])
        first=items[0] if items else None
        second=items[1] if len(items)>1 else None
        passed=bool(first is not None and second is not None and first.distance<.75*second.distance)
        ratio=first.distance/second.distance if first is not None and second is not None and second.distance>0 else None
        result.append({'direction':direction,'query_index':qi,'first_train_index':int(first.trainIdx) if first is not None else None,'first_distance':float(first.distance) if first is not None else None,'second_train_index':int(second.trainIdx) if second is not None else None,'second_distance':float(second.distance) if second is not None else None,'ratio':ratio,'passes_strict_ratio_075':passed})
    return result


def mutual_rows(forward,reverse):
    back={r['query_index']:r for r in reverse}
    result=[]
    for r in forward:
        if not r['passes_strict_ratio_075']:continue
        b=back.get(r['first_train_index'])
        if b is None or not b['passes_strict_ratio_075'] or b['first_train_index']!=r['query_index']:continue
        result.append({'match_index':len(result),'left_keypoint':r['query_index'],'right_keypoint':r['first_train_index'],'descriptor_l2':r['first_distance'],'forward_ratio':r['ratio'],'reverse_ratio':b['ratio']})
    return result


def summary(values,total_n):
    values=np.asarray(values,dtype=np.float64)
    finite=values[np.isfinite(values)]
    return {'total_n':int(total_n),'finite_n':int(len(finite)),'median':float(np.median(finite)) if len(finite) else None,'p95_linear':float(np.percentile(finite,95,method='linear')) if len(finite) else None}


def hull_coverage(points,shape_hw):
    points=np.asarray(points,dtype=np.float64).reshape(-1,2)
    valid=points[np.isfinite(points).all(axis=1)]
    area=float(cv2.contourArea(cv2.convexHull(valid.astype(np.float32)))) if len(valid)>=3 else 0.
    h,w=shape_hw
    return {'finite_points_n':len(valid),'hull_area_pixels_squared':area,'denominator_width_times_height':int(w*h),'fraction':area/(w*h)}


def calculate_residuals(left,right,H):
    n=len(left); inverse,condition,inverse_status=inverse_matrix(H)
    if H is None or inverse is None:
        blank=np.full(n,np.nan)
        return {'forward':blank.copy(),'backward':blank.copy(),'symmetric':blank.copy()},condition,inverse_status
    forward=np.linalg.norm(project(left,H)-right,axis=1)
    backward=np.linalg.norm(project(right,inverse)-left,axis=1)
    symmetric=np.sqrt((forward**2+backward**2)/2)
    return {'forward':forward,'backward':backward,'symmetric':symmetric},condition,inverse_status


def analyse_pair(left,right):
    forward=knn_rows(left['descriptors'],right['descriptors'],'left_to_right')
    reverse=knn_rows(right['descriptors'],left['descriptors'],'right_to_left')
    mutual=mutual_rows(forward,reverse)
    L=np.array([left['points_resized'][m['left_keypoint']] for m in mutual],np.float64).reshape(-1,2)
    R=np.array([right['points_resized'][m['right_keypoint']] for m in mutual],np.float64).reshape(-1,2)
    LO=np.array([left['points_original'][m['left_keypoint']] for m in mutual],np.float64).reshape(-1,2)
    RO=np.array([right['points_original'][m['right_keypoint']] for m in mutual],np.float64).reshape(-1,2)
    H=None;mask=None;error=None;status='insufficient_mutual_matches'
    if len(mutual)>=4:
        try:
            cv2.setRNGSeed(20261027)
            H,mask=cv2.findHomography(L,R,method=cv2.RANSAC,ransacReprojThreshold=3.,maxIters=2000,confidence=.995)
            status='homography_returned' if H is not None else 'homography_not_estimated'
        except cv2.error as exc:
            status='opencv_estimation_error';error=str(exc)
    if H is not None and not np.isfinite(H).all():
        status='homography_nonfinite';H=None
    inliers=mask.ravel().astype(bool) if mask is not None and len(mask)==len(mutual) else np.zeros(len(mutual),bool)
    H_original=np.linalg.inv(right['T_original_to_resized'])@H@left['T_original_to_resized'] if H is not None else None
    all_stats={};inlier_stats={};conditions={}
    for space,lp,rp,matrix in [('resized',L,R,H),('original',LO,RO,H_original)]:
        residuals,condition,inverse_status=calculate_residuals(lp,rp,matrix)
        conditions[space]={'condition_number':condition,'inverse_status':inverse_status}
        all_stats[space]={k:summary(v,len(mutual)) for k,v in residuals.items()}
        inlier_stats[space]={k:summary(v[inliers],int(inliers.sum())) for k,v in residuals.items()}
        for i,m in enumerate(mutual):
            m.update({space+'_left_x':safe(lp[i,0]),space+'_left_y':safe(lp[i,1]),space+'_right_x':safe(rp[i,0]),space+'_right_y':safe(rp[i,1])})
            m.update({space+'_'+k+'_error':safe(v[i]) for k,v in residuals.items()})
    for i,m in enumerate(mutual):m['ransac_inlier']=bool(inliers[i]) if mask is not None else None
    unique_joint=len(np.unique(np.c_[L[inliers],R[inliers]],axis=0)) if inliers.any() else 0
    result={'ransac_unique_inlier_xy_pairs_n':int(unique_joint),'status':status,'opencv_error':error,'truth_classification':None,'left_keypoints_n':len(left['keypoints']),'right_keypoints_n':len(right['keypoints']),'forward_ratio_pass_n':sum(r['passes_strict_ratio_075'] for r in forward),'reverse_ratio_pass_n':sum(r['passes_strict_ratio_075'] for r in reverse),'mutual_matches_n':len(mutual),'ransac_inliers_n':int(inliers.sum()),'ransac_inlier_fraction_denominator_mutual':float(inliers.mean()) if len(mutual) else None,'homography_left_to_right_resized':H.tolist() if H is not None else None,'homography_left_to_right_original':H_original.tolist() if H_original is not None else None,'ransac_mask':mask.ravel().astype(int).tolist() if mask is not None else None,'matrix_conditions':conditions,'residual_units':{'resized_forward':'right resized pixels','resized_backward':'left resized pixels','original_forward':'right original pixels','original_backward':'left original pixels','symmetric':'root mean squared forward/backward errors in respective stated pixel coordinate systems; descriptive'},'residuals_all_mutual':all_stats,'residuals_ransac_inliers':inlier_stats,'inlier_hull_coverage_resized':{'left':hull_coverage(L[inliers],left['resized_shape_hw']),'right':hull_coverage(R[inliers],right['resized_shape_hw'])},'inlier_hull_coverage_original':{'left':hull_coverage(LO[inliers],left['original_shape_hw']),'right':hull_coverage(RO[inliers],right['original_shape_hw'])},'no_binary_truth_threshold':True}
    return result,forward,reverse,mutual


def synthetic_grid_check(H_fitted,H_true,source_shape_hw,valid_output_mask):
    h,w=source_shape_hw
    grid=np.array([(x*(w-1),y*(h-1)) for y in np.linspace(.1,.9,9) for x in np.linspace(.1,.9,9)])
    expected=project(grid,H_true);estimated=project(grid,H_fitted) if H_fitted is not None else np.full_like(expected,np.nan)
    oh,ow=valid_output_mask.shape
    valid=np.isfinite(expected).all(axis=1)&(expected[:,0]>=0)&(expected[:,0]<=ow-1)&(expected[:,1]>=0)&(expected[:,1]<=oh-1)
    for i in np.flatnonzero(valid):
        x,y=np.rint(expected[i]).astype(int)
        valid[i]=valid_output_mask[y,x]>0
    errors=np.linalg.norm(expected-estimated,axis=1)
    rows=[{'grid_index':i,'source_x':float(p[0]),'source_y':float(p[1]),'truth_right_x':safe(expected[i,0]),'truth_right_y':safe(expected[i,1]),'estimated_right_x':safe(estimated[i,0]),'estimated_right_y':safe(estimated[i,1]),'valid_expected_overlap':bool(valid[i]),'projection_error_right_original_pixels':safe(errors[i])} for i,p in enumerate(grid)]
    return {'all_grid_n':len(grid),'valid_overlap_n':int(valid.sum()),'error_on_valid_expected_overlap':summary(errors[valid],int(valid.sum())),'role':'fixed grid positions not fitting correspondences; not independent held-out scenarios'},rows

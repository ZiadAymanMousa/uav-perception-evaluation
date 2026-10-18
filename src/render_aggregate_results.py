# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Render tables/figures from bundled aggregate scores; never performs inference or scoring."""
import csv,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/matplotlib_config'))

def main():
    d=ROOT/'results';a=json.loads((d/'aggregate_results.json').read_text());c=json.loads((d/'current_matcher_results.json').read_text())
    methods=a['method_order']; protocols=a['protocol_order']
    with (d/'three_seed_protocol_scores.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['seed','protocol','method','ap50_95_percent','slicing_minus_full960_pp'])
        for seed in a['training_seeds']:
            for i,proto in enumerate(protocols):
                for m in methods:w.writerow([seed,proto,m,a['scores_ap50_95_percent'][seed][i][m],a['slicing_minus_full960_pp'][seed][i]])
    fields=['seed','method','matcher','cap_mode','integration','ap50_percent','ap50_95_percent']
    with (d/'current_matcher_scores.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(c['scores'])
    fields=['seed','method','ap50_percent','ap50_95_percent','first_sweep_median_ms','first_sweep_p95_ms','detector_passes','processed_input_pixels','processed_input_pixel_ratio_vs_full640']
    with (d/'development_operating_points.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in a['development_methods']:
            w.writerow(dict(seed=row['seed'],method=row['method'],ap50_percent=row['ap50_percent'],ap50_95_percent=row['ap50_95_percent'],first_sweep_median_ms=row['diagnostic_median_ms'],first_sweep_p95_ms=row['diagnostic_p95_ms'],detector_passes=row['total_detector_passes'],processed_input_pixels=row['total_processed_input_pixels'],processed_input_pixel_ratio_vs_full640=row['processed_input_pixel_ratio_vs_full640']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(15,5.4),constrained_layout=True)
    labels={'full_640':'Full 640','full_960':'Full 960','full_1280':'Full 1280','fixed_slices_640_plus_global':'Slices + global'}
    for m in methods:axes[0].plot([0,1,2],[a['scores_ap50_95_percent'][s][0][m] for s in [0,1,2]],marker='o',label=labels[m])
    axes[0].set(title='Pinned converted protocol',xlabel='Training seed',ylabel='AP50-95 (%)',xticks=[0,1,2]);axes[0].legend(fontsize=8)
    short=['Pinned / 1000','Pinned / 500','Legacy macro','Legacy aggregate','AP boundary / 1000','AP boundary / 500']
    for s in [0,1,2]:axes[1].plot(range(6),a['slicing_minus_full960_pp'][s],marker='o',label='Seed '+str(s))
    axes[1].axhline(0,color='black',lw=.8);axes[1].set(title='Slicing minus full 960',ylabel='AP50-95 difference (pp)',xticks=range(6));axes[1].set_xticklabels(short,rotation=42,ha='right',fontsize=8);axes[1].legend(fontsize=8)
    x=np.arange(4)
    for s,style in [(0,'-'),(1,'--')]:
        for integ,color in [('pinned_ap_integration','#1565c0'),('reviewed_publisher_ap_boundary','#d95f02')]:
            vals=[]
            for m in methods:
                scores={r['matcher']:r['ap50_95_percent'] for r in c['scores'] if r['seed']==s and r['method']==m and r['cap_mode']=='all_saved' and r['integration']==integ}
                vals.append(scores['upstream_abc2066']-scores['pinned_8_3_253'])
            axes[2].plot(x,vals,style,marker='o',color=color,label='Seed '+str(s)+' / '+('pinned AP' if integ=='pinned_ap_integration' else 'AP boundary'))
    axes[2].axhline(0,color='black',lw=.8);axes[2].set(title='Captured matcher change: seeds 0 and 1',ylabel='Current minus pinned matcher (pp)',xticks=x);axes[2].set_xticklabels([labels[m] for m in methods],rotation=35,ha='right',fontsize=8);axes[2].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('548 exposed development images per model | known scoring controls | descriptive, not final-test or significance evidence',fontsize=10)
    for ext in ['png','svg']:fig.savefig(d/('three_seed_comparison.'+ext),dpi=180)
    plt.close(fig)
if __name__=='__main__':main()

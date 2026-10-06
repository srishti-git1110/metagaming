"""Render observed counts, not model-generated numbers, from saved Inspect logs."""
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .grader_inference import export_records,CONDITIONS
from .grader_report import metric

NAMES={'baseline':'No examples','marker_rewarded':'Marker rewarded','plain_rewarded':'Plain rewarded',
       'indifferent':'Equal scores','unscored':'No scores'}


def plot(directory):
    rows=export_records(directory/'experiment')
    if not rows:raise ValueError('No experiment samples')
    ids=sorted({r['prompt_id'] for r in rows})
    counts=defaultdict(list); checks={}
    for r in rows:
        if r['branch']=='behavior':counts[(r['prompt_id'],r['condition'])].append(metric(r,'marker'))
        else:checks[(r['prompt_id'],r['condition'])]=metric(r,'comprehension')
    matrix=np.full((len(ids),len(CONDITIONS)),np.nan)
    labels={}
    for i,pid in enumerate(ids):
        for j,c in enumerate(CONDITIONS):
            valid=[v for v in counts[(pid,c)] if v is not None]
            if valid:matrix[i,j]=sum(valid)/len(valid)
            labels[i,j]=f'{sum(valid)}/{len(valid)}' if valid else 'Missing'
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[5,3]},layout='constrained')
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#dddddd')
    ax.imshow(matrix,cmap=cmap,vmin=0,vmax=1,aspect='auto')
    for (i,j),label in labels.items():ax.text(j,i,label,ha='center',va='center',color='white' if matrix[i,j]>.6 else '#152c38',fontsize=13)
    ax.set_xticks(range(5),[NAMES[c] for c in CONDITIONS],rotation=25,ha='right')
    ax.set_yticks(range(len(ids)),ids);ax.set_title('Final answers with assigned prefix',loc='left',pad=15)
    cs=['marker_rewarded','plain_rewarded','indifferent']
    bx.set_xlim(-.5,2.5);bx.set_ylim(len(ids)-.5,-.5)
    for i,pid in enumerate(ids):
        for j,c in enumerate(cs):
            v=checks.get((pid,c));t='Pass' if v==1 else 'Fail' if v==0 else 'Missing'
            bx.text(j,i,t,ha='center',va='center',fontsize=11,color='#12605b' if v==1 else '#953524')
    bx.set_xticks(range(3),[NAMES[c] for c in cs],rotation=25,ha='right');bx.set_yticks([])
    bx.set_title('Separate grader prediction: 4/4 required',loc='left',pad=15)
    for sp in bx.spines.values():sp.set_visible(False)
    model=rows[0]['target_model'].replace('openrouter/','')
    fig.suptitle(f'{model}: observed prefix counts and comprehension',fontsize=16)
    out=directory/'analysis';out.mkdir(exist_ok=True)
    fig.savefig(out/'prefix-and-comprehension.png',dpi=180)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    plot(p.parse_args().directory)

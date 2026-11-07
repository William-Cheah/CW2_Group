"""Reproducible sensitivity analysis using the supplied NaiveBelief JSONL.
Only uses numpy/pandas; brute-force Euclidean distance equals sqrt(IndexFlatL2 distance).
Run: python threshold_sensitivity.py '/path/to/eval_local_severe.jsonl'
"""
import json, sys
import numpy as np
import pandas as pd
from pathlib import Path

MIN_IMPACT=3.0
HARM_REWARD=-3.0
TAU_EXIT=0.07
OUTCOME_WINDOW=8


def vector(inc):
    o=np.asarray(inc['start_obs'],dtype=np.float32)
    return np.r_[o[:7],o[7:9],np.float32(min(inc['start_belief_var']/0.25,1.0)),o[19:34:4]].astype(np.float32)


def find_incidents(df):
    df=df.sort_values(['episode','step']).reset_index(drop=True).copy()
    cutoff=df.belief_var.quantile(0.90)
    df['is_incident']=(df.temp_breach.astype(bool)|(df.units_wasted>0)|df.override.astype(bool)|(df.fallback_level>0)|(df.belief_var>cutoff))
    df['block']=((df.is_incident!=df.is_incident.shift())|(df.episode!=df.episode.shift())).cumsum()
    out=[]
    for block,part in df[df.is_incident].groupby('block',sort=True):
        first,last=part.iloc[0],part.iloc[-1]
        after=df[(df.episode==first.episode)&(df.step>last.step)&(df.step<=last.step+OUTCOME_WINDOW)]
        out.append(dict(episode=int(first.episode),start_step=int(first.step),start_obs=first.obs,start_belief_var=float(first.belief_var),end_belief_var=float(after.belief_var.iloc[-1]) if len(after) else float(first.belief_var),override=bool(part.override.any()),resolved=(len(after)==OUTCOME_WINDOW),outcome_reward=float(part.reward.sum()+after.reward.sum()),outcome_wasted=int(part.units_wasted.sum()+after.units_wasted.sum()),outcome_breach=bool(part.temp_breach.any() or after.temp_breach.any())))
    return out


def eligible(inc):
    return inc['resolved'] and abs(inc['outcome_reward'])>=MIN_IMPACT


def label(inc):
    return 'harmful' if inc['outcome_reward']<=HARM_REWARD or inc['outcome_wasted']>0 or inc['outcome_breach'] else 'good'


def gate(incidents,threshold):
    vectors=[];records=[];counts={'stored':0,'merged':0,'contested':0,'rejected_insignificant':0,'rejected_unresolved':0}
    for inc in incidents:
        if not inc['resolved']:
            counts['rejected_unresolved']+=1; continue
        if abs(inc['outcome_reward'])<MIN_IMPACT:
            counts['rejected_insignificant']+=1;continue
        v=vector(inc);lab=label(inc)
        if vectors:
            ds=np.linalg.norm(np.vstack(vectors)-v,axis=1)
            i=int(np.argmin(ds))
            if ds[i]<=threshold: # IncidentDB.search uses <= on max_distance
                if records[i]['label']==lab:
                    records[i]['count']+=1;counts['merged']+=1
                else:
                    records[i]['contested']=True
                    vectors.append(v);records.append({'label':lab,'count':1,'contested':True})
                    counts['contested']+=1
                continue
        vectors.append(v);records.append({'label':lab,'count':1,'contested':False});counts['stored']+=1
    return np.asarray(vectors,dtype=np.float32).reshape(-1,14),records,counts


def main(path):
    df=pd.DataFrame(json.loads(line) for line in Path(path).open() if line.strip())
    inc=find_incidents(df)
    print('run',df['run'].unique().tolist(),'rows',len(df),'episodes',df.episode.nunique(),'incidents',len(inc))
    for thr in [0.1,0.2,0.3]:
        vs,rs,c=gate(inc,thr)
        print('dup',thr,'records',len(rs),c)
    train=[x for x in inc if x['episode']<15]
    test=[x for x in inc if x['episode']>=15 and eligible(x)]
    vs,rs,c=gate(train,0.2)
    print('retrieval train records',len(rs),'test eligible',len(test))
    for radius in [0.3,0.5,0.7]:
        matched=0;total=0;detail=[]
        for inc1 in test:
            if not len(vs):continue
            ds=np.linalg.norm(vs-vector(inc1),axis=1)
            order=np.argsort(ds)[:3]
            d=[float(ds[i]) for i in order if ds[i]<=radius]
            if d: matched+=1
            total+=len(d)
            detail.append((inc1['episode'],inc1['start_step'],[round(x,3) for x in d]))
        print('radius',radius,'query matches',matched,'/',len(test),'returned',total)


if __name__=='__main__':
    main(sys.argv[1] if len(sys.argv)>1 else 'logs/local/eval_local_severe.jsonl')

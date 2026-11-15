"""Explainable, diversity-aware recommendations from a CSV music catalog."""
import argparse, csv, json
from pathlib import Path
import numpy as np
FEATURES = ['energy','danceability','acousticness','valence','tempo']

def recommend(rows, seeds, k=8, diversity=0.25, weights=None):
    if not 0 <= diversity <= 1: raise ValueError('diversity must be between 0 and 1')
    if k < 1: raise ValueError('k must be positive')
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate track IDs')
    seeds = list(dict.fromkeys(seeds))
    if not seeds or any(s not in ids for s in seeds): raise ValueError('Provide known seed IDs')
    x = np.array([[float(r[f]) for f in FEATURES] for r in rows])
    if not np.isfinite(x).all(): raise ValueError('Features must be finite')
    w = np.ones(len(FEATURES)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(FEATURES),) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Provide five finite nonnegative weights, at least one positive')
    # Fit scale on catalog; constant features contribute zero distance.
    z = (x-x.mean(axis=0))/np.where(x.std(axis=0)>0,x.std(axis=0),1)
    z = z*np.sqrt(w/w.sum())
    seed_idx = [ids.index(s) for s in seeds]
    profile = z[seed_idx].mean(axis=0)
    distances = np.linalg.norm(z-profile,axis=1)
    relevance = 1/(1+distances)
    selected=[]
    candidates=[i for i in range(len(rows)) if i not in seed_idx]
    while candidates and len(selected)<k:
        def score(i):
            redundancy=max((1/(1+np.linalg.norm(z[i]-z[j])) for j in selected),default=0)
            return (1-diversity)*relevance[i]-diversity*redundancy
        best=max(candidates,key=score)
        selected.append(best);candidates.remove(best)
    return [dict(rows[i], relevance=round(float(relevance[i]),4),
                 explanation='Closest profile matches: '+', '.join([FEATURES[j] for j in np.argsort((z[i]-profile)**2) if w[j]>0][:2])) for i in selected]

def load(path):
    with open(path,newline='') as f: return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',default=str(Path(__file__).with_name('data')/'demo_tracks.csv'))
    p.add_argument('--seeds',nargs='+',default=['track-000','track-001'])
    p.add_argument('-k',type=int,default=8)
    p.add_argument('--diversity',type=float,default=.25)
    p.add_argument('--weights',nargs=5,type=float)
    a=p.parse_args()
    try: print(json.dumps(recommend(load(a.catalog),a.seeds,a.k,a.diversity,a.weights),indent=2))
    except (ValueError,KeyError) as e: p.error(str(e))
if __name__=='__main__': main()

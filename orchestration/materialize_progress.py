#!/usr/bin/env python3
import argparse,json,os
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--root',required=True);p.add_argument('--method',required=True);a=p.parse_args()
d=json.load(open(a.manifest)); outroot=os.path.join(a.root,'results','zoomkv_aligned_20261007' if a.method=='zoomkv' else 'pariskv')
for task,t in d['tasks'].items():
    ds=t['dataset']; os.makedirs(os.path.join(outroot,ds),exist_ok=True)
    path=os.path.join(outroot,ds,task+'.jsonl')
    with open(path,'w',encoding='utf-8') as f:
        for r in t['rows']:
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(task,len(t['rows']),path)

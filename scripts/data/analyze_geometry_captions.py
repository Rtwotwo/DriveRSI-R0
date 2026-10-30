"""Descriptive pilot measurements; not an automatic visual admission test."""
from pathlib import Path
import argparse
import collections
import json
import math
import re
import statistics
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from driversi.data.catalog.bailian_caption import write
from driversi.rendering.whitebox.exporter import sha256


def rank(docs, query):
    terms=lambda t:re.findall(r'[a-z]{3,}', t.lower())
    bags=[collections.Counter(terms(t)) for t in docs]
    vocab=sorted(set(w for b in bags for w in b)|set(terms(query)))
    index={w:i for i,w in enumerate(vocab)}
    idf={w:math.log((len(docs)+1)/(1+sum(w in b for b in bags)))+1 for w in vocab}
    def vec(b):
        v=np.zeros(len(vocab))
        for w,c in b.items():v[index[w]]=(1+math.log(c))*idf[w]
        return v/max(float(np.linalg.norm(v)),1e-12)
    mat=np.array([vec(b)for b in bags]);q=vec(collections.Counter(terms(query)))
    scores=mat@q
    return sorted(range(len(docs)),key=lambda i:(-scores[i],i)), scores.tolist(), mat


def facets(features):
    predicates=[features['pedestrian_count_30m']>=2,features['bicycle_count_30m']>=1,
        features['vehicle_count_30m']>=8,features['vehicle_count_30m']<=2,
        features['ego_speed_mps']<.2,features['ego_speed_mps']>=8,
        features['history_yaw_range_rad']>=.08,features['nearby_static_count_15m']>=2]
    phrases=['pedestrian populated street','bicycle cyclist cycling','dense many vehicles traffic',
        'sparse few vehicles traffic','ego stopped stationary','ego fast driving speed',
        'curved turning bend road','nearby static roadside obstacles']
    return predicates,' '.join(s for b,s in zip(predicates,phrases)if b)


def analyze(root):
    root=Path(root);sel=json.loads((root/'selection.json').read_text());jobs=json.loads((root/'jobs.json').read_text())
    report=json.loads((root/'pilot-v3_1/report.json').read_text())
    results={(r['token'],r['scope']):json.loads(Path(r['file']).read_text()) for r in report['rows']if 'file'in r}
    documents=[];visual=[];truth=[];warnings=[];failure=[]
    for row in sel['rows']:
        h=results.get((row['token'],'retrieval_history'),{})
        text=' '.join(c['text'] for c in h.get('candidate_claims',{}).values())
        gold,structured=facets(row['descriptor']['features'])
        documents.append(structured);visual.append(text);truth.append(gold)
    protocol=json.loads((root/'retrieval-protocol.json').read_text())
    structural=[]
    for qi,q in enumerate(protocol['queries']):
        gold=[i for i,t in enumerate(truth)if t[qi]]
        rec={'id':q['id'],'positive_count':len(gold),'query':q['query'],'rankings':{}}
        for name,ds in [('faceted_source',documents),('appearance_candidates',visual),('combined',[a+' '+b for a,b in zip(documents,visual)])]:
            ids,scores,_=rank(ds,q['query']);rec['rankings'][name]={'precision_at_5':sum(i in gold for i in ids[:5])/5 if gold else None,
                'top5_tokens':[sel['rows'][i]['token']for i in ids[:5]],'top5_scores':[scores[i]for i in ids[:5]]}
        structural.append(rec)
    labels=json.loads((root/'visual-audit/retrieval-labels.json').read_text())
    appearance=[]
    for q in protocol['visual_queries']:
        eligible=[r for r in labels['rows']if r['labels'][q['id']]is not None]
        ids,scores,_=rank([visual[r['index']]for r in eligible],q['query'])
        k=min(5,len(ids));positive=sum(r['labels'][q['id']]for r in eligible)
        appearance.append({'id':q['id'],'query':q['query'],'eligible_scenes':len(eligible),'positive_count':positive,
            'precision_at_5':sum(eligible[i]['labels'][q['id']]for i in ids[:k])/k if positive else None,
            'recall_at_5':sum(eligible[i]['labels'][q['id']]for i in ids[:k])/positive if positive else None,
            'maximum_precision_at_5_given_positive_count':min(positive,k)/k,
            'top5_indices':[eligible[i]['index']for i in ids[:k]],'labels_scope':'anchor four-camera assistant audit only'})
    all_receipts=[]
    for folder in ['pilot','pilot-v3_1']:
        for p in (root/folder/'attempts').glob('*.json'):
            receipt=json.loads(p.read_text());all_receipts.append(receipt)
    total_in=sum(r.get('response',{}).get('usage',{}).get('prompt_tokens',0)for r in all_receipts)
    total_out=sum(r.get('response',{}).get('usage',{}).get('completion_tokens',0)for r in all_receipts)
    usages_unknown=sum(not r.get('response',{}).get('usage')for r in all_receipts)
    for (token,scope),r in results.items():
        warnings += [dict(token=token,scope=scope,**x)for x in r.get('quarantined_claims',[])]
        if r.get('status')!='format_valid_unreviewed':failure.append({'token':token,'scope':scope,'error':r.get('error'),'status':r.get('status')})
    valid=[r for r in results.values()if r.get('status')=='format_valid_unreviewed']
    lengths=[r['elapsed_seconds']for r in results.values()if isinstance(r.get('elapsed_seconds'),(int,float))]
    _,_,mat=rank(visual,'')
    pair=(mat@mat.T)[np.triu_indices(len(mat),1)]
    audits=[json.loads((Path(r['bundle'])/'audit.json').read_text())for r in sel['rows']]
    cost=(total_in*2+total_out*8)/1e6
    result={'scene_count':len(sel['rows']),'unique_source_groups':len({r['source_group']for r in sel['rows']}),
        'cities':dict(collections.Counter(r['city']for r in sel['rows'])),
        'final_scope_records':len(results),'final_format_valid':len(valid),'failures':failure,
        'whitebox_audits_passed':sum(a['passed']for a in audits),'whitebox_frames':sum(a['frames_checked']for a in audits),
        'actor_states_checked':sum(a['actor_states_checked']for a in audits),
        'api_requests_including_spine':len(all_receipts),'usage_unknown_calls':usages_unknown,
        'prompt_tokens_including_rejected':total_in,'completion_tokens_including_rejected':total_out,
        'published_price_estimate_rmb':{'busy':cost,'idle':cost/2,'source':'https://help.aliyun.com/zh/model-studio/model-pricing',
            'not_invoice':True,'excludes':'cache discounts, free quota, account-specific billing; unknown usage excluded'},
        'median_job_latency_seconds_includes_rate_limiter':statistics.median(lengths)if lengths else None,
        'structural_retrieval':structural,'appearance_retrieval_16_audit':appearance,
        'appearance_text_diversity':{'nonempty':sum(bool(x)for x in visual),'unique_fraction':len(set(x for x in visual if x))/max(1,sum(bool(x)for x in visual)),
            'mean_pairwise_tfidf_cosine':float(pair.mean()),'not_semantic_accuracy':True},
        'quarantined_claims':warnings,'automatic_appearance_approval':False,'generated_video_quality_measured':False,
        'geometry_authority_preserved':all(not r['released_prompts']['appearance_included']for r in valid),
        'caveats':protocol['limits']+['Facet retrieval uses the same source predicates as relevance labels and is a wiring check, not independent quality evidence.',
            'Appearance retrieval uses 16 assistant-reviewed anchor mosaics; no external adjudication or statistical generalization.',
            'Neither caption format nor lexical diversity proves factual correctness. VLM drafts remain unreviewed.',
            'This pilot exports object/camera trajectories, not full map/visibility conditions; complete_video_condition=false.']}
    write(root/'analysis.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    r=analyze(a.output);print(json.dumps({k:v for k,v in r.items()if k not in ['structural_retrieval','appearance_retrieval_16_audit','quarantined_claims','caveats']},indent=2))

"""Rejudge saved baseline reasoning with the original VEA rubric; no target inference."""
import argparse,json
from pathlib import Path
from dotenv import load_dotenv
from inspect_ai import score
from inspect_ai.log import read_eval_log
from .data import ROOT
from .scoring import vea
from .grader_inference import JUDGE,JUDGE_CONFIG
from .pilot import model_args


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path);p.add_argument('--run',action='store_true')
    args=p.parse_args()
    logs=[read_eval_log(x) for x in (args.directory/'experiment').glob('*.eval')]
    log=next(x for x in logs if x.eval.task=='inference_experiment_baseline_behavior')
    out=args.directory/'analysis'/'baseline-upstream-judge.json'
    print(json.dumps({'target_calls':0,'judge_calls':len(log.samples or []),'output':str(out),'run':args.run}))
    if not args.run:return
    if out.exists():raise SystemExit('Preserving existing rescoring output')
    load_dotenv(ROOT/'.env',override=False)
    result=score(log,[vea(JUDGE,'auto',model_args('azure'),JUDGE_CONFIG,tolerant=True)],action='append',display='plain')
    rows=[{'prompt_id':str(s.id),'epoch':s.epoch,'value':s.scores['vea'].value if 'vea' in (s.scores or {}) else None,
           'evidence':s.scores['vea'].metadata if 'vea' in (s.scores or {}) else None} for s in result.samples or []]
    out.write_text(json.dumps({'rubric':'original upstream OLMo VEA rubric, same as screening','target_inference_repeated':False,'rows':rows},indent=2)+'\n')
    (out.parent/'baseline-upstream-inspect-log.json').write_text(result.model_dump_json(indent=2)+'\n')
    print(json.dumps({'positive':sum(r['value']==1 for r in rows),'valid':sum(r['value'] is not None for r in rows),'rows':rows}))

if __name__=='__main__':main()

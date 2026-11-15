"""Eight-parent A1 full-solve score gate, paired with C0 on the same CPU."""
import argparse
import json
import os
import time
import traceback
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metric', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.environ['JACOBIK_NUM_WORKERS'] = '1'
    os.environ['JACOBIK_PROFILE'] = 'u1'
    os.environ['JACOBIK_PROFILE_OVERRIDES'] = json.dumps({'adam_metric':True} if args.metric else {})
    import torch
    torch.set_num_threads(1)
    from tools.jgen.run_g3_sanity import capture_parent
    from tools.jgen.run_o1_g2 import PARENTS
    from tools.jgen import inputs
    from tools.jdev.budget.common import stats_from_db
    stats = stats_from_db(inputs.DB_PATH)
    result = {'metric':args.metric,'parents':{},'failures':{}}
    out = Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    for code in PARENTS:
        t0 = time.monotonic()
        try:
            cap, solved = capture_parent(code,stats)
            result['parents'][code] = {'top1_score':float(solved.conformers[0].score),
                'rows':len(solved.conformers),'wall_s':time.monotonic()-t0}
        except Exception:
            result['failures'][code] = traceback.format_exc()
        out.write_text(json.dumps(result,indent=2))
        print(code,result['parents'].get(code,result['failures'].get(code)),flush=True)

if __name__ == '__main__':
    main()

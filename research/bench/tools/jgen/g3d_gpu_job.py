"""Run a paired bench invocation under the GPU lock and hard directive stops."""
import argparse,json,logging,subprocess,time
from pathlib import Path
from tools.runkit.resources import gpu_lock


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    parser.add_argument('--panel',required=True)
    parser.add_argument('--only',default='')
    parser.add_argument('--job-dir',type=Path,required=True)
    args=parser.parse_args();args.job_dir.mkdir(parents=True,exist_ok=True)
    smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True,check=True)
    (args.job_dir/'nvidia-smi.txt').write_text(smi.stdout)
    started=time.monotonic();info=dict(panel=args.panel,only=args.only,spec=args.spec)
    validity={};exempt=[]
    def stop(kind,message,parent=None):
        event=dict(kind=kind,message=message,parent=parent)
        (args.job_dir/'STOP.json').write_text(json.dumps(event,indent=2)+'\n')
        print('DIRECTIVE STOP',json.dumps(event),flush=True)
        raise SystemExit(86)
    class Guard(logging.Handler):
        def emit(self,record):
            message=record.getMessage()
            if 'step recording failed at step' in message:
                stop('replay_fallback',message,getattr(validity.get('instance'),'parent',None))
            if 'GFN2 embed refinement failed' in message:
                parent=getattr(validity.get('instance'),'parent',None)
                if parent=='NEPNEF':exempt.append(dict(parent=parent,embed_fallback=True,message=message))
                else:stop('nonexempt_embed_fallback',message,parent)
    from tools.jgen.embedding_validity import EmbeddingValidity
    original_install=EmbeddingValidity.install;original_check=EmbeddingValidity.check
    def install(item):
        validity['instance']=item
        return original_install(item)
    def check(item,parent):
        if parent in item.invalid and parent!='NEPNEF':
            stop('nonexempt_embed_fallback','GFN2 embedding fallback (including cached failure)',parent)
        return original_check(item,parent)
    EmbeddingValidity.install=install;EmbeddingValidity.check=check
    guard=Guard(logging.WARNING);logging.getLogger().addHandler(guard)
    try:
        with gpu_lock('CodexB'):
            import torch
            torch.use_deterministic_algorithms(True)
            from tools.jgen.bench.run import main as bench
            cmd=['--arms-spec',args.spec,'--panel',args.panel,'--seeds','11,12,13','--xyz-compression','xz']
            if args.only:cmd+=['--only',args.only]
            info['exit_code']=bench(cmd)
    except BaseException as exc:
        info['exception']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        info['wall_s']=time.monotonic()-started
        info['NEPNEF_exempt_fallbacks']=exempt
        (args.job_dir/'job.json').write_text(json.dumps(info,indent=2)+'\n')
        logging.getLogger().removeHandler(guard)
        EmbeddingValidity.install=original_install;EmbeddingValidity.check=original_check
        if exempt:
            for arm in json.loads(Path(args.spec).read_text()):
                for path in (Path(arm['out'])/'records').glob('NEPNEF_*.json'):
                    row=json.loads(path.read_text());row['embed_fallback']=True
                    path.write_text(json.dumps(row))

if __name__=='__main__':main()

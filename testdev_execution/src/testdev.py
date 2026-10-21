# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Prepare, verify, execute or rescore an explicit portable test-dev task."""
import argparse
import json
from pathlib import Path
from common import (load_task, require, write_json, read_json, digest, source_hashes,
                    offline_environment, runtime_context, now)
from data import prepare, verify
from checkpoints import checkpoint_context

def checkpoint_options(args):
    return dict(seed=args.seed, baseline_run=args.baseline_run,
                training_data=args.training_data, training_config=args.training_config,
                training_binding=args.training_binding)


def validate_paths(args):
    """Reject unsafe writes before configuring/importing any runtime."""
    protected = [args.raw_root.resolve(), args.prepared.resolve()]
    protected += [Path(p).resolve() for p in (args.training_data,args.baseline_run) if p is not None]
    def outside(path, label):
        value = Path(path).resolve()
        require(all(not value.is_relative_to(root) for root in protected), label + ' must be outside input data and baseline trees')
        return value
    operation = args.operation
    if operation == 'prepare':
        require(not args.prepared.resolve().is_relative_to(args.raw_root.resolve()), 'Preparation must be outside raw input tree')
    if operation in ('infer','rescore-log'):
        require(args.output is not None, 'Fresh explicit output required')
        outside(args.output,'Output')
        require(not args.output.exists(), 'Output exists; preserve it')
    else:
        require(args.output is None, 'Output argument only applies to infer and rescore-log')
    if operation == 'rescore-log':
        require(args.predictions is not None and args.predictions.is_file(), 'Existing saved prediction log required')
    if operation == 'verify-run':
        require(args.run is not None and args.run.is_dir() and not args.smoke, 'Existing explicit run required; scope is read from its record')
    if operation in ('prepare','verify','infer'):
        require(args.cache_root is None, 'Cache root is only used for preflight or saved-output scoring')
        return None
    default = (args.output.parent if args.output is not None else
               (args.run.parent if args.run is not None else args.prepared.parent))/'testdev_runtime'
    cache = outside(args.cache_root if args.cache_root is not None else default,'Runtime cache')
    require(not cache.exists() or cache.is_dir(), 'Runtime cache must be a directory')
    outside(cache/'runtime_settings','Runtime settings')
    outside(cache/'matplotlib_settings','Matplotlib settings')
    return cache

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['prepare','verify','preflight','infer','verify-run','rescore-log'])
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--weights',type=Path)
    parser.add_argument('--mode',choices=['frozen-replay','prospective-completed-baseline'])
    parser.add_argument('--seed',choices=['seed0','seed1','seed2'])
    parser.add_argument('--baseline-run',type=Path)
    parser.add_argument('--training-data',type=Path)
    parser.add_argument('--training-config',type=Path)
    parser.add_argument('--training-binding',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--cache-root',type=Path,help='Optional writable runtime cache outside all input/baseline trees')
    parser.add_argument('--run',type=Path)
    parser.add_argument('--predictions',type=Path)
    parser.add_argument('--smoke',action='store_true',help='Exactly the first sorted image on CPU; not complete confirmation')
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args()
    cache = validate_paths(args)
    task = load_task(args.config)
    if args.operation == 'prepare':
        require(not args.prepared.resolve().is_relative_to(args.raw_root.resolve()), 'Preparation must be outside raw input tree')
        result = prepare(args.raw_root,args.prepared,args.config,task)
    else:
        context = verify(args.raw_root,args.prepared,args.config,task)
        if args.operation == 'verify':
            result = {'status':'complete-preparation-verified',**context}
        else:
            require(args.weights is not None and args.mode is not None, 'Explicit weights and checkpoint mode required')
            options = checkpoint_options(args)
            checkpoint = checkpoint_context(task,args.weights,args.mode,**options)
            if args.operation == 'infer':
                require(args.output is not None, 'Fresh explicit output directory required')
                from infer import infer
                result = infer(args.raw_root,args.prepared,args.output,args.config,task,args.weights,
                               mode=args.mode,smoke=args.smoke,device=args.device,**options)
                result = {k:result[k] for k in ('status','scope','images_compared','prediction_log_sha256','summary_sha256')}
            else:
                # Use a user-selected writable cache location; no framework import at module load.
                offline_environment(cache)
                runtime = runtime_context(task,exact_inference=False)
                if args.operation == 'preflight':
                    result = {'status':'preflight-pass','data_context':context,'checkpoint_context':checkpoint,'runtime':runtime,
                              'note':'No inference, training or evaluator independence established'}
                else:
                    from records import score_log, verify_run_files
                    if args.operation == 'verify-run':
                        require(args.run is not None and not args.smoke, 'Explicit run required; scope read from verified run')
                        record, smoke = verify_run_files(args.run,task,context,checkpoint,prepared=args.prepared)
                        summary, accounting = score_log(args.run/'predictions_and_timings.jsonl',args.prepared,task,smoke=smoke)
                        require(summary == read_json(args.run/'summary.json'), 'Same-scorer recomputation differs from saved summary')
                        require(accounting == record['accounting'], 'Recomputed matrix accounting differs')
                        require(record['environment']['framework_source_hashes'] == runtime['framework_source_hashes'], 'Run framework hashes differ')
                        result = {'status':'completed-run-verified','scope':record['scope'],'same_pinned_scorer':True,
                                  'independent_evaluator':False,'accounting':accounting}
                    else:
                        require(args.predictions is not None and args.output is not None, 'Saved log and fresh output JSON required')
                        require(not args.output.exists(), 'Rescore output exists; preserve it')
                        require(not args.output.resolve().is_relative_to(args.raw_root.resolve()) and not args.output.resolve().is_relative_to(args.prepared.resolve()), 'Rescore output must be outside input data')
                        summary, accounting = score_log(args.predictions,args.prepared,task,smoke=args.smoke)
                        result = {'schema':'uvd1-testdev-saved-log-rescore-v1','status':'completed','created_utc':now(),
                                  'purpose':'Saved-log rescore only; no new inference or verification of original runtime by association',
                                  'same_pinned_scorer':True,'independent_evaluator':False,'data_context':context,
                                  'checkpoint_context':checkpoint,'runtime':runtime,'source_hashes':source_hashes(),
                                  'accounting':accounting,'summary':summary}
                        args.output.parent.mkdir(parents=True,exist_ok=True)
                        write_json(args.output,result)
                        result = {'status':'saved-log-rescored','output_sha256':digest(args.output),'accounting':accounting,
                                  'same_pinned_scorer':True,'independent_evaluator':False}
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()

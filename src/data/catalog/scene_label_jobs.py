"""One-scene review pilot using the authorized private visual-model client."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import bailian_caption as client
from . import scene_labels as labels
from . import scene_label_summary as summary
from .store import identity
from driversi.rendering.whitebox.conditioning import DIMENSION_SAMPLING
from driversi.rendering.whitebox.exporter import sha256
from driversi.rendering.whitebox.motion_audit import audit_source_motion


def pilot_config(private_file):
    config=client.private_config(private_file)
    if len(config['models']) != 1:
        raise ValueError('Review pilot needs one explicitly configured model')
    config.update(concurrency=1,max_requests=1,max_attempts=1,max_tokens=6000,image_max_side=960)
    return config


def prepare(database, bundle, config, output):
    output=Path(output);bundle=Path(bundle)
    source=json.loads((bundle/'source.json').read_text())
    if source.get('dimension_sampling')!=DIMENSION_SAMPLING:
        raise ValueError('Use current source whitebox sampling; preserve/re-export old bundle')
    if not audit_source_motion(source)['passed']:
        raise ValueError('Source motion audit failed')
    context=labels.measurements(source)
    context['fact_registry']=sorted(labels.fact_ids(context))
    context['taxonomy']=labels.taxonomy()
    context['response_schema']=labels.response_schema()
    prompt=(labels.CONFIG/'prompt.md').read_text()
    # An output directory represents ONE immutable input. Check before any writes.
    if (output/'job.json').exists():
        job=json.loads((output/'job.json').read_text())
        plan=json.loads((output/'execution-plan.json').read_text())
        if (job['source_sha256']!=sha256(bundle/'source.json') or
            Path(job['bundle']).resolve()!=(bundle/'source.json').resolve() or
            job['prompt']!=prompt or job['facts']!=context or
            job['image_max_side']!=config['image_max_side'] or
            job['jpeg_quality']!=config['jpeg_quality'] or plan['model']!=config['models'][0]):
            raise ValueError('Different scene/contract: retain this run and use a new output directory')
        validate_evidence(job)
        return job
    if (output/'sample.json').exists():
        raise ValueError('Orphan sample requires reconciliation, not overwrite')
    # Reuse the established photo preparation; only the full-scene job is sent.
    jobs=client.prepare(database,[bundle],config,output/'evidence')
    job=next(j for j in jobs if j['scope']=='video_clip')
    client.write(output/'context.json',context)
    job.update(scope=labels.SCOPE,prompt=prompt,
        facts=context,context_file=str((output/'context.json').resolve()),
        context_sha256=sha256(output/'context.json'))
    job['input_identity']=identity({k:v for k,v in job.items() if k!='input_identity'})
    client.write(output/'job.json',job)
    client.write(output/'execution-plan.json',dict(scene_count=1,model=config['models'][0],
        api_requests_cap=1,auto_retry=False,batch_annotation=False,online_input=False,
        source_split='navtrain',production_pool_write=False))
    return job


def validate_evidence(job):
    if sha256(job['bundle'])!=job['source_sha256'] or sha256(job['context_file'])!=job['context_sha256']:
        raise ValueError('Scene evidence changed after preparation')
    for e in job['evidence']:
        if sha256(e['file'])!=e['sha256'] or sha256(e['source_file'])!=e['source_sha256']:
            raise ValueError('Photo evidence changed after preparation')


def decode(job,model,response):
    choice=response['choices'][0]
    raw=choice['message']['content']
    if choice.get('finish_reason')!='stop' or not isinstance(raw,str):
        raise ValueError('Incomplete scene-label response')
    validate_evidence(job)
    context=json.loads(Path(job['context_file']).read_text())
    observations=labels.parse(raw,context,job['evidence'])
    source=json.loads(Path(job['bundle']).read_text())
    sample=labels.assemble(source,context,observations,job['evidence'],model)
    return dict(status='format_valid_unreviewed',raw=raw,usage=response.get('usage'),
                model=model,returned_model=response.get('model'),request_id=response.get('id'),
                scene_label=sample,trainable=False)


def archive_current_sample(output):
    """Keep prior primary views recoverable when their representation changes."""
    output=Path(output); old=output/'sample.json'
    if not old.exists():return None
    digest=sha256(old);prior='superseded/'+digest+'.json'
    destination=output/prior;destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():destination.write_bytes(old.read_bytes())
    elif sha256(destination)!=digest:raise ValueError('Prior sample archive changed')
    record_archive_base(output,digest)
    return prior


def record_archive_base(output, digest):
    """An archived primary file's relative links still use its original run root."""
    output=Path(output)
    archive=output/'superseded'/(''+digest+'.json')
    if sha256(archive)!=digest:raise ValueError('Archived sample hash mismatch')
    metadata=dict(archived_file=digest+'.json',sha256=digest,
                  original_file='sample.json',reference_base='..',
                  meaning='Resolve references inside archived_file against reference_base, relative to this metadata file.')
    path=archive.with_suffix('.references.json')
    if path.exists():
        if json.loads(path.read_text())!=metadata:raise ValueError('Archive reference metadata changed')
    else:client.write(path,metadata)


def publish(job, model, receipt, output):
    """A cached API success is not a current-parser publication success."""
    output=Path(output)
    status=dict(status='validating_current_contract',trainable=False,
                label_code_sha256=sha256(labels.__file__))
    client.write(output/'publication.json',status)
    try:
        refreshed=decode(job,model,dict(choices=[dict(finish_reason='stop',message=dict(content=receipt['raw']))],
            usage=receipt.get('usage'),model=receipt.get('returned_model'),id=receipt.get('request_id')))
        full=refreshed['scene_label']
        full['provenance']=dict(input_identity=job['input_identity'],source_sha256=job['source_sha256'],
            context_sha256=job['context_sha256'],request_identity=receipt['request_identity'],
            label_code_sha256=sha256(labels.__file__))
        # This full record now lives one level below the run directory.
        full['support_files']={k:'../'+v for k,v in full['support_files'].items()}
        # Full evidence is content-addressed and never overwritten with a new revision.
        relative='audit/labels-'+identity(full)+'.json'
        audit=output/relative
        if not audit.exists():client.write(audit,full)
        elif json.loads(audit.read_text())!=full:raise ValueError('Full label audit changed')
        sample=summary.compact(full,dict(path=relative,sha256=sha256(audit)))
    except Exception as exc:
        old=output/'sample.json'
        prior=archive_current_sample(output)
        # No exception message/raw provider content is persisted here.
        status.update(status='current_validation_rejected',error_type=type(exc).__name__,
                      previous_sample=prior,format_valid=False,approved_for_batch=False)
        client.write(old,dict(schema_version=labels.VERSION,review=status,trainable=False))
        client.write(output/'publication.json',status)
        raise
    archive_current_sample(output)
    client.write(output/'sample.json',sample)
    status.update(status='valid_current_contract_awaiting_user_review',format_valid=True,
                  approved_for_batch=False,sample_sha256=sha256(output/'sample.json'),
                  summary_code_sha256=sha256(summary.__file__),
                  summary_profile_sha256=sha256(summary.PROFILE),full_audit=relative)
    client.write(output/'publication.json',status)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',required=True)
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--private-config',required=True)
    parser.add_argument('--credential-file')
    parser.add_argument('--output',required=True)
    parser.add_argument('--execute',action='store_true',help='One real request; otherwise only prepare evidence')
    args=parser.parse_args()
    import fcntl
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'pilot.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(args)


def execute(args):
    config=pilot_config(args.private_config)
    job=prepare(args.database,args.bundle,config,args.output)
    if not args.execute:
        print(json.dumps(dict(status='prepared',scene_count=1,api_calls=0)));return
    if not args.credential_file:
        raise ValueError('--execute requires an explicit authorized private credential file')
    result=client.run([job],config,Path(args.output)/'api',key=client.credential(args.credential_file),decode=decode)
    if result['complete']:
        receipt=json.loads(Path(result['rows'][0]['file']).read_text())
        # Revalidate/rebuild cached raw output with CURRENT parser; no new inference.
        publish(job,config['models'][0],receipt,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    if not result['complete']:raise SystemExit(2)


if __name__=='__main__':main()

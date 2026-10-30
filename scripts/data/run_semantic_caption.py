"""Bounded local Qwen captions; structure remains authoritative and raw text unreviewed."""
import argparse
import json
import time
from pathlib import Path

from driversi.data.catalog.store import Catalog, identity
from driversi.data.catalog.condition_caption import assemble, parse, prompt_path
from driversi.data.catalog import scene_description
from driversi.data.catalog.labeling import LocalAnnotator
from driversi.data.sources.navsim import load_log
from driversi.rendering.whitebox.conditioning import CAMERAS, describe, write_json
from driversi.rendering.whitebox.exporter import sha256


def main():
    from PIL import Image
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database',required=True);p.add_argument('--bundle',type=Path,action='append',required=True)
    p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--schema-version',type=int,choices=[1,2],default=2)
    p.add_argument('--scope',choices=['retrieval_history','video_clip'],action='append')
    args=p.parse_args()
    if not 1<=len(args.bundle)<=4:raise ValueError('1..4 bundles per bounded pilot')
    args.output.mkdir(parents=True,exist_ok=True)
    cat=Catalog(args.database,'navtrain',readonly=True)
    config=dict(backend='qwen3_5',model_path=args.model,max_new_tokens=1536 if args.schema_version==2 else 512,max_seconds_per_scene=90)
    model_files={q.name:sha256(q) for q in Path(args.model).glob('*') if q.is_file() and q.suffix in ('.json','.safetensors','.jinja')}
    runtime=None;receipts=[]
    try:
        for bundle in args.bundle:
            source=json.loads((bundle/'source.json').read_text());record=cat.scene(source['token'])['record']
            raw_frames=load_log(Path(record['log_path']),trusted=True);a=record['anchor_index']
            for scope,indices in [('retrieval_history',[a-3,a-2,a-1,a]),('video_clip',[a-4,a,a+5,a+10])]:
                if args.scope and scope not in args.scope:continue
                facts=scene_description.scoped_measurements(source,scope) if args.schema_version==2 else describe(source,scope)
                selected_prompt=scene_description.prompt_path_v2(scope) if args.schema_version==2 else prompt_path(scope)
                prompt=selected_prompt.read_text()
                evidence=[];images=[]
                for j in indices:
                    row=raw_frames[j]
                    for cam in CAMERAS:
                        file=Path(record['sensor_root'])/row['cams'][cam]['data_path']
                        with Image.open(file) as im:
                            im=im.convert('RGB');im.thumbnail((384,384));images.append(im.copy())
                        evidence.append(dict(id=f'{cam}_f{j-a:+d}',camera=cam,
                                             relative_time_s=(int(row['timestamp'])-source['anchor_timestamp_us'])/1e6,
                                             file=str(file),sha256=sha256(file)))
                if scope=='retrieval_history' and any(x['relative_time_s']>0 for x in evidence):raise ValueError('Future caption leakage')
                adapter_sha256=sha256(__file__)
                compiler_sha256=sha256(scene_description.__file__)
                signature=identity(dict(facts=facts,evidence=evidence,prompt=prompt,model=model_files,config=config,adapter_sha256=adapter_sha256,
                                       schema_version=args.schema_version,compiler_sha256=compiler_sha256))
                out=args.output/f'{source["token"]}-{scope}.json'
                if out.exists():
                    old=json.loads(out.read_text())
                    if old['input_identity']!=signature:raise ValueError('Caption inputs changed; use new output version')
                    receipts.append(dict(file=out.name,status='reused',input_identity=signature));continue
                if runtime is None:runtime=LocalAnnotator(config)
                compact={k:v for k,v in facts.items() if k not in ('actor_presence','anchor_objects')}
                compact['nearest_annotated_objects']=facts['anchor_objects'][:12]
                content=[{'type':'text','text':prompt+'\nAuthoritative structured context:\n'+json.dumps(compact)}]
                for row,im in zip(evidence,images,strict=True):
                    content += [{'type':'text','text':f"Image {row['id']} t={row['relative_time_s']:.6f}s"},{'type':'image','image':im}]
                content.append({'type':'text','text':('Return the fixed v2 JSON now. IMPORTANT: every field is a JSON OBJECT {"text":"description","evidence_ids":["exact image ID"]}, NEVER a plain string. distinctive_features is an array of those objects. Choose ONLY1or2 supporting image IDs per claim; NEVER copy all16. At most2 distinctive features. Available IDs: ' if args.schema_version==2 else 'Return the five-field JSON now. For distinctive_details include one exact image ID: ')+', '.join(x['id'] for x in evidence)})
                messages=[{'role':'user','content':content}]
                prefix=''
                if args.schema_version==2:
                    prefix='{"overview": {"text": "'
                    messages.append({'role':'assistant','content':[{'type':'text','text':prefix}]})
                inputs=runtime.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=args.schema_version==1,
                          continue_final_message=args.schema_version==2,return_dict=True,return_tensors='pt',enable_thinking=False).to('cuda')
                started=time.monotonic()
                with torch.inference_mode():generated=runtime.model.generate(**inputs,do_sample=False,max_new_tokens=config['max_new_tokens'],max_time=90)
                raw=prefix+runtime.processor.decode(generated[0,inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
                error=None;appearance=None;format_repair=None
                try:
                    if args.schema_version==2:appearance,format_repair=scene_description.parse_generation(raw,[x['id'] for x in evidence])
                    else:appearance=parse(raw)
                    if args.schema_version==1 and appearance['distinctive_details'].lower()!='unknown' and not any(x['id'] in appearance['distinctive_details'] for x in evidence):
                        raise ValueError('Distinctive details omit an exact source-image evidence ID')
                except Exception as exc:error=type(exc).__name__+': '+str(exc)
                caption=scene_description.build_description(source,scope,observations=appearance if error is None else None,evidence=evidence) if args.schema_version==2 else assemble(facts,appearance if error is None else None)
                value=dict(token=source['token'],scope=scope,input_identity=signature,evidence=evidence,prompt_sha256=sha256(selected_prompt),
                           model_files=model_files,config=config,adapter_sha256=adapter_sha256,raw=raw,error=error,elapsed_seconds=time.monotonic()-started,
                           compiler_sha256=compiler_sha256,schema_version=args.schema_version,caption=caption,format_valid=error is None,format_repair=format_repair,
                           factual_quality='unreviewed',trainable=False)
                write_json(out,value);receipts.append(dict(file=out.name,status='format_valid_unreviewed' if error is None else 'rejected',input_identity=signature))
                write_json(args.output/'report.json',dict(rows=receipts,complete=False))
                print(json.dumps(receipts[-1]),flush=True)
        write_json(args.output/'report.json',dict(rows=receipts,complete=True))
    finally:cat.close()


if __name__=='__main__':main()

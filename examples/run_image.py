# SPDX-License-Identifier: Apache-2.0
"""Single-image greedy generation with VAPOR; no benchmark data required."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['qwen','internvl'],default='qwen')
    p.add_argument('--model',required=True,help='Local checkpoint directory or Hugging Face model ID')
    p.add_argument('--revision',help='Checkpoint/code revision; pin a commit for reproducibility')
    p.add_argument('--image',required=True,type=Path)
    p.add_argument('--prompt',default='Describe the image and explain the visual evidence for your answer.')
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--rank',type=int,default=16)
    p.add_argument('--fixed-threshold',type=float,help='Disable consistency calibration')
    p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--output',type=Path,help='Optional new JSON file; existing files are not overwritten')
    return p.parse_args()

def main():
    a=parse_args()
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoTokenizer, AutoModel, Qwen3VLForConditionalGeneration
    from vapor import VAPORConfig
    torch.set_num_threads(1); torch.manual_seed(0)
    config=(VAPORConfig() if a.family=='qwen' else VAPORConfig.internvl())
    config=replace(config,rank=a.rank,max_new_tokens=a.max_new_tokens)
    if a.fixed_threshold is not None:
        config=replace(config,threshold=a.fixed_threshold,calibrate=False)
    load=dict(local_files_only=a.local_files_only)
    if a.revision: load['revision']=a.revision
    with Image.open(a.image) as im: image=im.convert('RGB')
    if a.family=='qwen':
        from vapor.adapters.qwen import vapor
        processor=AutoProcessor.from_pretrained(a.model,**load)
        model=Qwen3VLForConditionalGeneration.from_pretrained(
            a.model,dtype=torch.bfloat16,attn_implementation='eager',**load).to('cuda:0').eval()
        model.config.vision_config._attn_implementation='sdpa'
        model.model.visual.config._attn_implementation='sdpa'
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':a.prompt}]}]
        rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        batch=processor(text=[rendered],images=[image],return_tensors='pt')
        input_ids=batch['input_ids']; batch=batch.to('cuda:0')
        with torch.inference_mode():
            with vapor(model,processor.tokenizer,input_ids,config) as state:
                generated=model.generate(**batch,max_new_tokens=config.max_new_tokens,
                                         do_sample=False,use_cache=True,cache_implementation='dynamic')
            ids=generated[0,input_ids.shape[1]:].tolist(); stats=state.summary()
        answer=processor.tokenizer.decode(ids,skip_special_tokens=True)
    else:
        from vapor.adapters.internvl import vapor
        from vapor.adapters.vision import install_vision_sdpa
        from internvl_image import prepare_image
        tokenizer=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,use_fast=False,**load)
        fast=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,use_fast=True,**load)
        model=AutoModel.from_pretrained(a.model,torch_dtype=torch.bfloat16,
                    trust_remote_code=True,use_flash_attn=False,**load).to('cuda:0').eval()
        install_vision_sdpa(model)
        model.img_context_token_id=tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        batch,rendered,eos,sep=prepare_image(model,tokenizer,image,a.prompt)
        input_ids=batch['input_ids']
        batch={k:v.to('cuda:0',dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda:0') for k,v in batch.items()}
        with torch.inference_mode():
            with vapor(model,fast,input_ids,rendered,config) as state:
                generated=model.generate(**batch,max_new_tokens=config.max_new_tokens,
                    do_sample=False,use_cache=True,cache_implementation='dynamic',
                    eos_token_id=eos,return_dict_in_generate=True)
            ids=generated.sequences[0].tolist(); stats=state.summary()
        answer=tokenizer.decode(ids,skip_special_tokens=True).split(sep)[0].strip()
    result=dict(answer=answer,generated_tokens=len(ids),vapor=stats,
                model=a.model,revision=a.revision,family=a.family)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if a.output:
        with a.output.open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)

if __name__=='__main__':main()

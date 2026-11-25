# SPDX-License-Identifier: Apache-2.0
"""Image preparation following InternVL's documented tile/template convention.

This small public example is separate from benchmark-specific input readers.
"""
import importlib

def tile_image(image, limit=12, size=448):
    from PIL import Image
    w,h=image.size
    grids=sorted({(c,r) for n in range(1,limit+1) for c in range(1,n+1)
                  for r in range(1,n+1) if c*r<=limit},key=lambda cr:cr[0]*cr[1])
    best=(1,1); error=float('inf')
    for cols,rows in grids:
        d=abs(w/h-cols/rows)
        if d<error or (d==error and w*h>.5*size*size*cols*rows):
            best=(cols,rows);error=d
    cols,rows=best
    # PIL's default resize convention matches the native tiling helper.
    canvas=image.resize((cols*size,rows*size))
    tiles=[canvas.crop((c*size,r*size,(c+1)*size,(r+1)*size))
           for r in range(rows) for c in range(cols)]
    if len(tiles)>1:tiles.append(image.resize((size,size)))
    return tiles

def prepare_image(model, tokenizer, image, prompt):
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    transform=T.Compose([T.Resize((448,448),interpolation=InterpolationMode.BICUBIC),
                         T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tiles=tile_image(image.convert('RGB'))
    pixels=torch.stack([transform(tile) for tile in tiles])
    module=importlib.import_module(type(model).__module__)
    template=module.get_conv_template(model.template)
    template.system_message=model.system_message
    template.append_message(template.roles[0],'<image>\n'+prompt)
    template.append_message(template.roles[1],None)
    rendered=template.get_prompt().replace('<image>',
            '<img>'+'<IMG_CONTEXT>'*(model.num_image_token*len(tiles))+'</img>',1)
    inputs=tokenizer(rendered,return_tensors='pt')
    sep=template.sep.strip()
    return dict(pixel_values=pixels,input_ids=inputs.input_ids,attention_mask=inputs.attention_mask),rendered,tokenizer.convert_tokens_to_ids(sep),sep

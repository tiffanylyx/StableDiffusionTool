
import argparse
import torch
from diffusers import StableDiffusionPipeline,StableDiffusionImg2ImgPipeline
import requests
from io import BytesIO
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--imgprompt",
        type=str,
        nargs="?",
        default=None,
        help="the image prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/"
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="?",
        help="dir to write results to",
        default=100
    )
    parser.add_argument(
        "--strength",
        type=int,
        nargs="?",
        help="strength of style",
        default=0.5
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="?",
        help="random seed",
        default=None
    )
    parser.add_argument(
        "--scale",
        type=int,
        nargs="?",
        help="guidance_scale",
        default=7.5
    )  
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="number of images generate once",
    )
    
    opt = parser.parse_args()
    text_prompt = opt.prompt
    image_prompt_path = opt.imgprompt 
    height = opt.H
    width = opt.W
    num_inference_steps = opt.steps
    strength = opt.strength
    guidance_scale = opt.scale
    seed = opt.seed
    num_images = opt.batch
    path = opt.outdir
    
    
    model_version = 'CompVis/stable-diffusion-v1-4'
    if image_prompt_path==None:
    	pipe = StableDiffusionPipeline.from_pretrained(model_version, revision="fp16", torch_dtype=torch.float16).to("cuda")
    else:
    	pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_version,revision="fp16",torch_dtype=torch.float16).to("cuda")
    if seed != None:
        generator = torch.Generator("cuda").manual_seed(seed)
    else:
        generator = torch.Generator("cuda")
    prompt_list = [text_prompt] * num_images


    if image_prompt_path==None:
        images = pipe(prompt_list, height = height, width = width,num_inference_steps = num_inference_steps,guidance_scale = guidance_scale, strength = strength,generator=generator).images
    else:
        if image_prompt_path.startswith("/"):
            init_image = Image.open(image_prompt_path)
        else:
    	    response = requests.get(image_prompt_path)
    	    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    	    init_image = init_image.resize((width, height))
        images = pipe(prompt=prompt_list, init_image=init_image, height = height, width = width,num_inference_steps = num_inference_steps,guidance_scale = guidance_scale,  strength = strength,generator=generator).images


    grid = image_grid(images, rows=1, cols=num_images)


    for index,image in enumerate(images):
        image.save(path+"/"+prompt_list[index]+"_"+str(height)+"*"+str(width)+"_"+str(seed)+"_"+str(guidance_scale)+"_"+str(index)+".jpg")
  	
main()


import os
import gc
import cv2
import requests
import numpy as np
import gradio as gr
import torch
import traceback
from tqdm import tqdm
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Define URLs and their corresponding local storage paths
face_model = {
    "GFPGANv1.2.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth",
    "GFPGANv1.3.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "RestoreFormer.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
    "CodeFormer.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/CodeFormer.pth",
}
realesr_model = {
    "realesr-general-x4v3.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "realesr-animevideov3.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRNet_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "4x-AnimeSharp.pth": "https://huggingface.co/utnah/esrgan/resolve/main/4x-AnimeSharp.pth?download=true",
}
files_to_download = [
    ( "a1.jpg",
      "https://thumbs.dreamstime.com/b/tower-bridge-traditional-red-bus-black-white-colors-view-to-tower-bridge-london-black-white-colors-108478942.jpg" ),
    ( "a2.jpg",
      "https://media.istockphoto.com/id/523514029/photo/london-skyline-b-w.jpg?s=612x612&w=0&k=20&c=kJS1BAtfqYeUDaORupj0sBPc1hpzJhBUUqEFfRnHzZ0=" ),
    ( "a3.jpg",
      "https://i.guim.co.uk/img/media/06f614065ed82ca0e917b149a32493c791619854/0_0_3648_2789/master/3648.jpg?width=700&quality=85&auto=format&fit=max&s=05764b507c18a38590090d987c8b6202" ),
    ( "a4.jpg",
      "https://i.pinimg.com/736x/46/96/9e/46969eb94aec2437323464804d27706d--victorian-london-victorian-era.jpg" ),
]

# Ensure the target directory exists
os.makedirs("weights", exist_ok=True)
os.makedirs('output', exist_ok=True)

def download_from_url(output_path, url):
    try:
        # Check if the file already exists
        if os.path.exists(output_path):
            print(f"File already exists, skipping download: {output_path}")
            return

        print(f"Downloading: {url}")
        with requests.get(url, stream=True) as response, open(output_path, "wb") as f:
            total_size = int(response.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"Download successful: {output_path}")
    except requests.RequestException as e:
        print(f"Download failed: {url}, Error: {e}")


# Iterate through each file
for output_path, url in files_to_download:
    # Check if the file already exists
    if os.path.exists(output_path):
        print(f"File already exists, skipping download: {output_path}")
        continue

    # Start downloading
    download_from_url(output_path, url)


def inference(img, version, realesr, scale: float):
    print(img, version, scale)
    try:
        img_name = os.path.basename(str(img))
        basename, extension = os.path.splitext(img_name)
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        elif len(img.shape) == 2:  # for gray inputs
            img_mode = None
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_mode = None

        h, w = img.shape[0:2]
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
            
        if version:
            download_from_url(os.path.join("weights", version), face_model[version])
        if realesr:
            download_from_url(os.path.join("weights", realesr), realesr_model[realesr])

        # background enhancer with RealESRGAN
        if realesr == 'RealESRGAN_x4plus.pth':  # x4 RRDBNet model
            netscale = 4
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale)
        elif realesr == 'RealESRNet_x4plus.pth':  # x4 RRDBNet model
            netscale = 4
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale)
        elif realesr == 'RealESRGAN_x4plus_anime_6B.pth':  # x4 RRDBNet model with 6 blocks
            netscale = 4
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=netscale)
        elif realesr == 'RealESRGAN_x2plus.pth':  # x2 RRDBNet model
            netscale = 2
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale)
        elif realesr == 'realesr-animevideov3.pth':  # x4 VGG-style model (XS size)
            netscale = 4
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=netscale, act_type='prelu')
        elif realesr == 'realesr-general-x4v3.pth':  # x4 VGG-style model (S size)
            netscale = 4
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=netscale, act_type='prelu')
        # elif realesr == '4x-AnimeSharp.pth':  # 4x-AnimeSharp
        #     netscale = 4
        #     model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale)
        
        half = True if torch.cuda.is_available() else False
        upsampler = RealESRGANer(scale=netscale, model_path=os.path.join("weights", realesr), model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

        face_enhancer = None
        if version == 'GFPGANv1.2.pth':
            face_enhancer = GFPGANer(
            model_path='weights/GFPGANv1.2.pth', upscale=scale, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'GFPGANv1.3.pth':
            face_enhancer = GFPGANer(
            model_path='weights/GFPGANv1.3.pth', upscale=scale, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'GFPGANv1.4.pth':
            face_enhancer = GFPGANer(
            model_path='weights/GFPGANv1.4.pth', upscale=scale, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'RestoreFormer.pth':
            face_enhancer = GFPGANer(
            model_path='weights/RestoreFormer.pth', upscale=scale, arch='RestoreFormer', channel_multiplier=2, bg_upsampler=upsampler)
        elif version == 'CodeFormer.pth':
             face_enhancer = GFPGANer(
             model_path='weights/CodeFormer.pth', upscale=scale, arch='CodeFormer', channel_multiplier=2, bg_upsampler=upsampler)
             
        files = []
        outputs = []
        try:
            if face_enhancer:
                cropped_faces, restored_aligned, restored_img = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                # save faces
                if cropped_faces and restored_aligned:
                    for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_aligned)):
                        # save cropped face
                        save_crop_path = f"output/{basename}{idx:02d}_cropped_faces.png"
                        cv2.imwrite(save_crop_path, cropped_face)
                        # save restored face
                        save_restore_path = f"output/{basename}{idx:02d}_restored_faces.png"
                        cv2.imwrite(save_restore_path, restored_face)
                        # save comparison image
                        save_cmp_path = f"output/{basename}{idx:02d}_cmp.png"
                        cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                        cv2.imwrite(save_cmp_path, cmp_img)
                
                        files.append(save_crop_path)
                        files.append(save_restore_path)
                        files.append(save_cmp_path)
                        outputs.append(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
                        outputs.append(cv2.cvtColor(restored_face, cv2.COLOR_BGR2RGB))
                        outputs.append(cv2.cvtColor(cmp_img, cv2.COLOR_BGR2RGB))
            else:
                restored_img, _ = upsampler.enhance(img, outscale=scale)
        except RuntimeError as error:
            print(traceback.format_exc())
            print('Error', error)
        finally:
            if face_enhancer:
                face_enhancer._cleanup()
            else:
                # Free GPU memory and clean up resources
                torch.cuda.empty_cache()
                gc.collect()


        try:
            if scale != 2:
                interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                h, w = img.shape[0:2]
                restored_img = cv2.resize(restored_img, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
        except Exception as error:
            print(traceback.format_exc())
            print("wrong scale input.", error)

        if not extension:
            extension = ".png" if img_mode == "RGBA" else ".jpg" # RGBA images should be saved in png format
        save_path = f"output/{basename}{extension}"
        cv2.imwrite(save_path, restored_img)

        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        files.append(save_path)
        outputs.append(restored_img)
        return outputs, files
    except Exception as error:
        print(traceback.format_exc())
        print("global exception", error)
        return None, None


title = "Image Upscaling & Restoration(esp. Face) using GFPGAN Algorithm"
description = r"""Gradio demo for <a href='https://github.com/TencentARC/GFPGAN' target='_blank'><b>GFPGAN: Towards Real-World Blind Face Restoration and Upscalling of the image with a Generative Facial Prior</b></a>.<br>
Practically the algorithm is used to restore your **old photos** or improve **AI-generated faces**.<br>
To use it, simply just upload the concerned image.<br>
"""
article = r"""
[![download](https://img.shields.io/github/downloads/TencentARC/GFPGAN/total.svg)](https://github.com/TencentARC/GFPGAN/releases)
[![GitHub Stars](https://img.shields.io/github/stars/TencentARC/GFPGAN?style=social)](https://github.com/TencentARC/GFPGAN)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2101.04061)
<center><img src='https://visitor-badge.glitch.me/badge?page_id=dj_face_restoration_GFPGAN' alt='visitor badge'></center>
"""
demo = gr.Interface(
    inference, [
        gr.Image(type="filepath", label="Input", format="png"),
        gr.Dropdown(["GFPGANv1.2.pth",
                     "GFPGANv1.3.pth",
                     "GFPGANv1.4.pth",
                     "RestoreFormer.pth",
                     # "CodeFormer.pth",
                     None], type="value", value='GFPGANv1.4.pth', label='Face Restoration version', info="Face Restoration and RealESR can be freely combined in different ways, or one can be set to \"None\" to use only the other model. Face Restoration is primarily used for face restoration in real-life images, while RealESR serves as a background restoration model."),
        gr.Dropdown(["realesr-general-x4v3.pth",
                     "realesr-animevideov3.pth",
                     "RealESRGAN_x4plus_anime_6B.pth",
                     "RealESRGAN_x2plus.pth",
                     "RealESRNet_x4plus.pth",
                     "RealESRGAN_x4plus.pth",
                     # "4x-AnimeSharp.pth",
                     None], type="value", value='realesr-general-x4v3.pth', label='RealESR version'),
        gr.Number(label="Rescaling factor", value=2),
        # gr.Slider(0, 100, label='Weight, only for CodeFormer. 0 for better quality, 100 for better identity', value=50)
    ], [
        gr.Gallery(type="numpy", label="Output (The whole image)", format="png"),
        gr.File(label="Download the output image")
    ],
    title=title,
    description=description,
    article=article,
    examples=[['a1.jpg', 'GFPGANv1.4.pth', "realesr-general-x4v3.pth", 2], 
              ['a2.jpg', 'GFPGANv1.4.pth', "realesr-general-x4v3.pth", 2], 
              ['a3.jpg', 'GFPGANv1.4.pth', "realesr-general-x4v3.pth", 2],
              ['a4.jpg', 'GFPGANv1.4.pth', "realesr-general-x4v3.pth", 2]])
    
demo.queue(default_concurrency_limit=4)
demo.launch(inbrowser=True)
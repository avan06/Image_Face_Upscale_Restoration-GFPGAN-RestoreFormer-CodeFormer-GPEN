﻿import os
import gc
import re
import cv2
import numpy as np
import gradio as gr
import torch
import traceback
import math
from collections import defaultdict
from facexlib.utils.misc import download_from_url
from basicsr.utils.realesrganer import RealESRGANer


# Define URLs and their corresponding local storage paths
face_models = {
    "GFPGANv1.4.pth"      : ["https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                            "https://github.com/TencentARC/GFPGAN/", 
"""GFPGAN: Towards Real-World Blind Face Restoration and Upscalling of the image with a Generative Facial Prior.
GFPGAN aims at developing a Practical Algorithm for Real-world Face Restoration.
It leverages rich and diverse priors encapsulated in a pretrained face GAN (e.g., StyleGAN2) for blind face restoration."""],

    "RestoreFormer++.ckpt": ["https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer++.ckpt",
                            "https://github.com/wzhouxiff/RestoreFormerPlusPlus", 
"""RestoreFormer++: Towards Real-World Blind Face Restoration from Undegraded Key-Value Pairs.
RestoreFormer++ is an extension of RestoreFormer. It proposes to restore a degraded face image with both fidelity and \
realness by using the powerful fully-spacial attention mechanisms to model the abundant contextual information in the face and \
its interplay with reconstruction-oriented high-quality priors."""],

    "CodeFormer.pth"      : ["https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
                            "https://github.com/sczhou/CodeFormer", 
"""CodeFormer: Towards Robust Blind Face Restoration with Codebook Lookup Transformer (NeurIPS 2022).
CodeFormer is a Transformer-based model designed to tackle the challenging problem of blind face restoration, where inputs are often severely degraded.
By framing face restoration as a code prediction task, this approach ensures both improved mapping from degraded inputs to outputs and the generation of visually rich, high-quality faces.
"""],

    "GPEN-BFR-512.pth"    : ["https://huggingface.co/akhaliq/GPEN-BFR-512/resolve/main/GPEN-BFR-512.pth",
                            "https://github.com/yangxy/GPEN", 
"""GPEN: GAN Prior Embedded Network for Blind Face Restoration in the Wild.
GPEN addresses blind face restoration (BFR) by embedding a GAN into a U-shaped DNN, combining GAN’s ability to generate high-quality images with DNN’s feature extraction.
This design reconstructs global structure, fine details, and backgrounds from degraded inputs.
Simple yet effective, GPEN outperforms state-of-the-art methods, delivering realistic results even for severely degraded images."""],

    "GPEN-BFR-1024.pt"    : ["https://www.modelscope.cn/models/iic/cv_gpen_image-portrait-enhancement-hires/resolve/master/pytorch_model.pt",
                            "https://www.modelscope.cn/models/iic/cv_gpen_image-portrait-enhancement-hires/files", 
"""The same as GPEN but for 1024 resolution."""],

    "GPEN-BFR-2048.pt"    : ["https://www.modelscope.cn/models/iic/cv_gpen_image-portrait-enhancement-hires/resolve/master/pytorch_model-2048.pt",
                            "https://www.modelscope.cn/models/iic/cv_gpen_image-portrait-enhancement-hires/files", 
"""The same as GPEN but for 2048 resolution."""],

    # legacy model
    "GFPGANv1.3.pth"    : ["https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                          "https://github.com/TencentARC/GFPGAN/", "The same as GFPGAN but legacy model"],
    "GFPGANv1.2.pth"    : ["https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth",
                          "https://github.com/TencentARC/GFPGAN/", "The same as GFPGAN but legacy model"],
    "RestoreFormer.ckpt": ["https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer.ckpt",
                          "https://github.com/wzhouxiff/RestoreFormerPlusPlus", "The same as RestoreFormer++ but legacy model"],
}
upscale_models = {
    # SRVGGNet
    "realesr-general-x4v3.pth": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                                "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.3.0", 
"""add realesr-general-x4v3 and realesr-general-wdn-x4v3. They are very tiny models for general scenes, and they may more robust. But as they are tiny models, their performance may be limited."""],

    "realesr-animevideov3.pth": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
                                "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0", 
"""update the RealESRGAN AnimeVideo-v3 model, which can achieve better results with a faster inference speed."""],
    
    "4xLSDIRCompact.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompact/4xLSDIRCompact.pth",
                                "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompact", 
"""Phhofm: Upscale small good quality photos to 4x their size. This is my first ever released self-trained sisr upscaling model."""],
     
    "4xLSDIRCompactC.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompactC/4xLSDIRCompactC.pth",
                                "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompactC", 
"""Phhofm: 4x photo upscaler that handler jpg compression. Trying to extend my previous model to be able to handle compression (JPG 100-30) by manually altering the training dataset, since 4xLSDIRCompact cant handle compression. Use this instead of 4xLSDIRCompact if your photo has compression (like an image from the web)."""],
         
    "4xLSDIRCompactR.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompactC/4xLSDIRCompactR.pth",
                                "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompactC", 
"""Phhofm: 4x photo uspcaler that handles jpg compression, noise and slight. Extending my last 4xLSDIRCompact model to Real-ESRGAN, meaning trained on synthetic data instead to handle more kinds of degradations, it should be able to handle compression, noise, and slight blur."""],

    "4xLSDIRCompactN.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompact3/4xLSDIRCompactC3.pth",
                                "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompact3", 
"""Phhofm: Upscale good quality input photos to x4 their size. The original 4xLSDIRCompact a bit more trained, cannot handle degradation.
I am releasing the Series 3 from my 4xLSDIRCompact models. In general my suggestion is, if you have good quality input images use 4xLSDIRCompactN3, otherwise try 4xLSDIRCompactC3 which will be able to handle jpg compression and a bit of blur, or then 4xLSDIRCompactCR3, which is an interpolation between C3 and R3 to be able to handle a bit of noise additionally."""],

    "4xLSDIRCompactC3.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompact3/4xLSDIRCompactC3.pth",
                                "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompact3", 
"""Phhofm: Upscale compressed photos to x4 their size. Able to handle JPG compression (30-100).
I am releasing the Series 3 from my 4xLSDIRCompact models. In general my suggestion is, if you have good quality input images use 4xLSDIRCompactN3, otherwise try 4xLSDIRCompactC3 which will be able to handle jpg compression and a bit of blur, or then 4xLSDIRCompactCR3, which is an interpolation between C3 and R3 to be able to handle a bit of noise additionally."""],

    "4xLSDIRCompactR3.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompact3/4xLSDIRCompactR3.pth",
                                "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompact3", 
"""Phhofm: Upscale (degraded) photos to x4 their size. Trained on synthetic data, meant to handle more degradations.
I am releasing the Series 3 from my 4xLSDIRCompact models. In general my suggestion is, if you have good quality input images use 4xLSDIRCompactN3, otherwise try 4xLSDIRCompactC3 which will be able to handle jpg compression and a bit of blur, or then 4xLSDIRCompactCR3, which is an interpolation between C3 and R3 to be able to handle a bit of noise additionally."""],

    "4xLSDIRCompactCR3.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompact3/4xLSDIRCompactCR3.pth",
                                "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompact3", 
"""Phhofm: I am releasing the Series 3 from my 4xLSDIRCompact models. In general my suggestion is, if you have good quality input images use 4xLSDIRCompactN3, otherwise try 4xLSDIRCompactC3 which will be able to handle jpg compression and a bit of blur, or then 4xLSDIRCompactCR3, which is an interpolation between C3 and R3 to be able to handle a bit of noise additionally."""],

    "2xParimgCompact.pth": ["https://github.com/Phhofm/models/releases/download/2xParimgCompact/2xParimgCompact.pth",
                                "https://github.com/Phhofm/models/releases/tag/2xParimgCompact", 
"""Phhofm: A 2x photo upscaling compact model based on Microsoft's ImagePairs. This was one of the earliest models I started training and finished it now for release. As can be seen in the examples, this model will affect colors."""],

    "1xExposureCorrection_compact.pth": ["https://github.com/Phhofm/models/releases/download/1xExposureCorrection_compact/1xExposureCorrection_compact.pth",
                                "https://github.com/Phhofm/models/releases/tag/1xExposureCorrection_compact", 
"""Phhofm: This model is meant as an experiment to see if compact can be used to train on photos to exposure correct those using the pixel, perceptual, color, color and ldl losses. There is no brightness loss. Still it seems to kinda work."""],
    
    "1xUnderExposureCorrection_compact.pth": ["https://github.com/Phhofm/models/releases/download/1xExposureCorrection_compact/1xUnderExposureCorrection_compact.pth",
                                "https://github.com/Phhofm/models/releases/tag/1xExposureCorrection_compact", 
"""Phhofm: This model is meant as an experiment to see if compact can be used to train on underexposed images to exposure correct those using the pixel, perceptual, color, color and ldl losses. There is no brightness loss. Still it seems to kinda work."""],
    
    "1xOverExposureCorrection_compact.pth": ["https://github.com/Phhofm/models/releases/download/1xExposureCorrection_compact/1xOverExposureCorrection_compact.pth",
                                "https://github.com/Phhofm/models/releases/tag/1xExposureCorrection_compact", 
"""Phhofm: This model is meant as an experiment to see if compact can be used to train on overexposed images to exposure correct those using the pixel, perceptual, color, color and ldl losses. There is no brightness loss. Still it seems to kinda work."""],

    # RRDBNet
    "RealESRGAN_x4plus_anime_6B.pth": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                                      "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.2.4", 
"""We add RealESRGAN_x4plus_anime_6B.pth, which is optimized for anime images with much smaller model size. More details and comparisons with waifu2x are in anime_model.md"""],

    "RealESRGAN_x2plus.pth"         : ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                                      "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.1", 
"""Add RealESRGAN_x2plus.pth model"""],

    "RealESRNet_x4plus.pth"         : ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
                                      "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.1", 
"""This release is mainly for storing pre-trained models and executable files."""],

    "RealESRGAN_x4plus.pth"         : ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                                      "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0", 
"""This release is mainly for storing pre-trained models and executable files."""],

    # ESRGAN(oldRRDB)
    "4x-AnimeSharp.pth": ["https://huggingface.co/utnah/esrgan/resolve/main/4x-AnimeSharp.pth?download=true",
                         "https://openmodeldb.info/models/4x-AnimeSharp", 
"""Interpolation between 4x-UltraSharp and 4x-TextSharp-v0.5. Works amazingly on anime. It also upscales text, but it's far better with anime content."""],

    "4x_IllustrationJaNai_V1_ESRGAN_135k.pth": ["https://drive.google.com/uc?export=download&confirm=1&id=1qpioSqBkB_IkSBhEAewSSNFt6qgkBimP",
                                               "https://openmodeldb.info/models/4x-IllustrationJaNai-V1-DAT2", 
"""Purpose: Illustrations, digital art, manga covers
Model for color images including manga covers and color illustrations, digital art, visual novel art, artbooks, and more. 
DAT2 version is the highest quality version but also the slowest. See the ESRGAN version for faster performance."""],

    "2x-sudo-RealESRGAN.pth": ["https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/2x-sudo-RealESRGAN.pth",
                               "https://openmodeldb.info/models/2x-sudo-RealESRGAN", 
"""Pretrained: Pretrained_Model_G: RealESRGAN_x4plus_anime_6B.pth / RealESRGAN_x4plus_anime_6B.pth (sudo_RealESRGAN2x_3.332.758_G.pth)
Tried to make the best 2x model there is for drawings. I think i archived that. 
And yes, it is nearly 3.8 million iterations (probably a record nobody will beat here), took me nearly half a year to train. 
It can happen that in one edge is a noisy pattern in edges. You can use padding/crop for that. 
I aimed for perceptual quality without zooming in like 400%. Since RealESRGAN is 4x, I downscaled these images with bicubic."""],
    
    "2x-sudo-RealESRGAN-Dropout.pth": ["https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/2x-sudo-RealESRGAN-Dropout.pth",
                               "https://openmodeldb.info/models/2x-sudo-RealESRGAN-Dropout", 
"""Pretrained: Pretrained_Model_G: RealESRGAN_x4plus_anime_6B.pth / RealESRGAN_x4plus_anime_6B.pth (sudo_RealESRGAN2x_3.332.758_G.pth)
Tried to make the best 2x model there is for drawings. I think i archived that. 
And yes, it is nearly 3.8 million iterations (probably a record nobody will beat here), took me nearly half a year to train. 
It can happen that in one edge is a noisy pattern in edges. You can use padding/crop for that. 
I aimed for perceptual quality without zooming in like 400%. Since RealESRGAN is 4x, I downscaled these images with bicubic."""],

    "4xNomos2_otf_esrgan.pth": ["https://github.com/Phhofm/models/releases/download/4xNomos2_otf_esrgan/4xNomos2_otf_esrgan.pth",
                               "https://github.com/Phhofm/models/releases/tag/4xNomos2_otf_esrgan", 
"""Phhofm: Restoration, 4x ESRGAN model for photography, trained using the Real-ESRGAN otf degradation pipeline."""],

    "4xNomosWebPhoto_esrgan.pth": ["https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_esrgan/4xNomosWebPhoto_esrgan.pth",
                               "https://github.com/Phhofm/models/releases/tag/4xNomosWebPhoto_esrgan", 
"""Phhofm: Restoration, 4x ESRGAN model for photography, trained with realistic noise, lens blur, jpg and webp re-compression.
ESRGAN version of 4xNomosWebPhoto_RealPLKSR, trained on the same dataset and in the same way."""],

    # DATNet
    "4xNomos8kDAT.pth"                     : ["https://github.com/Phhofm/models/releases/download/4xNomos8kDAT/4xNomos8kDAT.pth",
                                             "https://openmodeldb.info/models/4x-Nomos8kDAT", 
"""Phhofm: A 4x photo upscaler with otf jpg compression, blur and resize, trained on musl's Nomos8k_sfw dataset for realisic sr, this time based on the DAT arch, as a finetune on the official 4x DAT model."""],

    "4x-DWTP-DS-dat2-v3.pth"               : ["https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/4x-DWTP-DS-dat2-v3.pth",
                                             "https://openmodeldb.info/models/4x-DWTP-DS-dat2-v3", 
"""DAT descreenton model, designed to reduce discrepancies on tiles due to too much loss of the first version, while getting rid of the removal of paper texture"""],

    "4xBHI_dat2_real.pth"                  : ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_real/4xBHI_dat2_real.pth",
                                             "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_real", 
"""Phhofm: 4x dat2 upscaling model for web and realistic images. It handles realistic noise, some realistic blur, and webp and jpg (re)compression. Trained on my BHI dataset (390'035 training tiles) with degraded LR subset."""],

    "4xBHI_dat2_otf.pth"                   : ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_otf/4xBHI_dat2_otf.pth",
                                             "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_otf", 
"""Phhofm: 4x dat2 upscaling model, trained with the real-esrgan otf pipeline on my bhi dataset. Handles noise and compression."""],

    "4xBHI_dat2_multiblur.pth"             : ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_multiblurjpg/4xBHI_dat2_multiblur.pth",
                                             "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_multiblurjpg", 
"""Phhofm: 4x dat2 upscaling model, trained with down_up,linear, cubic_mitchell, lanczos, gauss and box scaling algos, some average, gaussian and anisotropic blurs and jpg compression. Trained on my BHI sisr dataset."""],

    "4xBHI_dat2_multiblurjpg.pth"          : ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_multiblurjpg/4xBHI_dat2_multiblurjpg.pth",
                                             "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_multiblurjpg", 
"""Phhofm: 4x dat2 upscaling model, trained with down_up,linear, cubic_mitchell, lanczos, gauss and box scaling algos, some average, gaussian and anisotropic blurs and jpg compression. Trained on my BHI sisr dataset."""],

    "4x_IllustrationJaNai_V1_DAT2_190k.pth": ["https://drive.google.com/uc?export=download&confirm=1&id=1qpioSqBkB_IkSBhEAewSSNFt6qgkBimP",
                                             "https://openmodeldb.info/models/4x-IllustrationJaNai-V1-DAT2", 
"""Purpose: Illustrations, digital art, manga covers
Model for color images including manga covers and color illustrations, digital art, visual novel art, artbooks, and more. 
DAT2 version is the highest quality version but also the slowest. See the ESRGAN version for faster performance."""],

    "4x-PBRify_UpscalerDAT2_V1.pth": ["https://github.com/Kim2091/Kim2091-Models/releases/download/4x-PBRify_UpscalerDAT2_V1/4x-PBRify_UpscalerDAT2_V1.pth",
                                      "https://github.com/Kim2091/Kim2091-Models/releases/tag/4x-PBRify_UpscalerDAT2_V1", 
"""Kim2091: Yet another model in the PBRify_Remix series. This is a new upscaler to replace the previous 4x-PBRify_UpscalerSIR-M_V2 model.
This model far exceeds the quality of the previous, with far more natural detail generation and better reconstruction of lines and edges."""],

    "4xBHI_dat2_otf_nn.pth": ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_otf_nn/4xBHI_dat2_otf_nn.pth",
                              "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_otf_nn", 
"""Phhofm: 4x dat2 upscaling model, trained with the real-esrgan otf pipeline but without noise, on my bhi dataset. Handles resizes, and jpg compression."""],

    # HAT
    "4xNomos8kSCHAT-L.pth"  : ["https://github.com/Phhofm/models/releases/download/4xNomos8kSCHAT/4xNomos8kSCHAT-L.pth",
                              "https://openmodeldb.info/models/4x-Nomos8kSCHAT-L", 
"""Phhofm: 4x photo upscaler with otf jpg compression and blur, trained on musl's Nomos8k_sfw dataset for realisic sr. Since this is a big model, upscaling might take a while."""],

    "4xNomos8kSCHAT-S.pth"  : ["https://github.com/Phhofm/models/releases/download/4xNomos8kSCHAT/4xNomos8kSCHAT-S.pth",
                              "https://openmodeldb.info/models/4x-Nomos8kSCHAT-S", 
"""Phhofm: 4x photo upscaler with otf jpg compression and blur, trained on musl's Nomos8k_sfw dataset for realisic sr. HAT-S version/model."""],

    "4xNomos8kHAT-L_otf.pth": ["https://github.com/Phhofm/models/releases/download/4xNomos8kHAT-L_otf/4xNomos8kHAT-L_otf.pth",
                              "https://openmodeldb.info/models/4x-Nomos8kHAT-L-otf", 
"""Phhofm: 4x photo upscaler trained with otf"""],

    "4xBHI_small_hat-l.pth": ["https://github.com/Phhofm/models/releases/download/4xBHI_small_hat-l/4xBHI_small_hat-l.pth",
                              "https://github.com/Phhofm/models/releases/tag/4xBHI_small_hat-l", 
"""Phhofm: 4x hat-l upscaling model for good quality input. This model does not handle any degradations.
This model is rather soft, I tried to balance sharpness and faithfulness/non-artifacts.
For a bit sharper output, but can generate a bit of artifacts, you can try the 4xBHI_small_hat-l_sharp version,
also included in this release, which might still feel soft if you are used to sharper outputs."""],

    # RealPLKSR_dysample
    "4xHFA2k_ludvae_realplksr_dysample.pth": ["https://github.com/Phhofm/models/releases/download/4xHFA2k_ludvae_realplksr_dysample/4xHFA2k_ludvae_realplksr_dysample.pth",
                                             "https://openmodeldb.info/models/4x-HFA2k-ludvae-realplksr-dysample", 
"""Phhofm: A Dysample RealPLKSR 4x upscaling model for anime single-image resolution."""],

    "4xArtFaces_realplksr_dysample.pth"    : ["https://github.com/Phhofm/models/releases/download/4xArtFaces_realplksr_dysample/4xArtFaces_realplksr_dysample.pth",
                                             "https://openmodeldb.info/models/4x-ArtFaces-realplksr-dysample", 
"""Phhofm: A Dysample RealPLKSR 4x upscaling model for art / painted faces."""],

    "4x-PBRify_RPLKSRd_V3.pth"             : ["https://github.com/Kim2091/Kim2091-Models/releases/download/4x-PBRify_RPLKSRd_V3/4x-PBRify_RPLKSRd_V3.pth",
                                             "https://github.com/Kim2091/Kim2091-Models/releases/tag/4x-PBRify_RPLKSRd_V3", 
"""Kim2091: This update brings a new upscaling model, 4x-PBRify_RPLKSRd_V3. This model is roughly 8x faster than the current DAT2 model, while being higher quality. 
It produces far more natural detail, resolves lines and edges more smoothly, and cleans up compression artifacts better.
As a result of those improvements, PBR is also much improved. It tends to be clearer with less defined artifacts."""],

    "4xNomos2_realplksr_dysample.pth"      : ["https://github.com/Phhofm/models/releases/download/4xNomos2_realplksr_dysample/4xNomos2_realplksr_dysample.pth",
                                             "https://openmodeldb.info/models/4x-Nomos2-realplksr-dysample", 
"""Phhofm: A Dysample RealPLKSR 4x upscaling model that was trained with / handles jpg compression down to 70 on the Nomosv2 dataset, preserves DoF.
This model affects / saturate colors, which can be counteracted a bit by using wavelet color fix, as used in these examples."""],

    # RealPLKSR
    "2x-AnimeSharpV2_RPLKSR_Sharp.pth": ["https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV2_Set/2x-AnimeSharpV2_RPLKSR_Sharp.pth",
                                        "https://github.com/Kim2091/Kim2091-Models/releases/tag/2x-AnimeSharpV2_Set", 
"""Kim2091: This is my first anime model in years. Hopefully you guys can find a good use-case for it.
RealPLKSR (Higher quality, slower) Sharp: For heavily degraded sources. Sharp models have issues depth of field but are best at removing artifacts
"""],

    "2x-AnimeSharpV2_RPLKSR_Soft.pth" : ["https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV2_Set/2x-AnimeSharpV2_RPLKSR_Soft.pth",
                                         "https://github.com/Kim2091/Kim2091-Models/releases/tag/2x-AnimeSharpV2_Set", 
"""Kim2091: This is my first anime model in years. Hopefully you guys can find a good use-case for it.
RealPLKSR (Higher quality, slower) Soft: For cleaner sources. Soft models preserve depth of field but may not remove other artifacts as well"""],

    "4xPurePhoto-RealPLSKR.pth"       : ["https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/4xPurePhoto-RealPLSKR.pth",
                                        "https://openmodeldb.info/models/4x-PurePhoto-RealPLSKR", 
"""Skilled in working with cats, hair, parties, and creating clear images.
Also proficient in resizing photos and enlarging large, sharp images.
Can effectively improve images from small sizes as well (300px at smallest on one side, depending on the subject).
Experienced in experimenting with techniques like upscaling with this model twice and \
then reducing it by 50% to enhance details, especially in features like hair or animals."""],

    "2x_Text2HD_v.1-RealPLKSR.pth"    : ["https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/2x_Text2HD_v.1-RealPLKSR.pth",
                                        "https://openmodeldb.info/models/2x-Text2HD-v-1", 
"""Purpose: Upscale text in very low quality to normal quality.
The upscale model is specifically designed to enhance lower-quality text images, \
improving their clarity and readability by upscaling them by 2x.
It excels at processing moderately sized text, effectively transforming it into high-quality, legible scans.
However, the model may encounter challenges when dealing with very small text, \
as its performance is optimized for text of a certain minimum size. For best results, \
input images should contain text that is not excessively small."""],

    "2xVHS2HD-RealPLKSR.pth"          : ["https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/2xVHS2HD-RealPLKSR.pth",
                                        "https://openmodeldb.info/models/2x-VHS2HD", 
"""An advanced VHS recording model designed to enhance video quality by reducing artifacts such as haloing, ghosting, and noise patterns.
Optimized primarily for PAL resolution (NTSC might work good as well)."""],

    "4xNomosWebPhoto_RealPLKSR.pth"   : ["https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_RealPLKSR/4xNomosWebPhoto_RealPLKSR.pth",
                                        "https://openmodeldb.info/models/4x-NomosWebPhoto-RealPLKSR", 
"""Phhofm: 4x RealPLKSR model for photography, trained with realistic noise, lens blur, jpg and webp re-compression."""],

    # DRCT
    "4xNomos2_hq_drct-l.pth"          : ["https://github.com/Phhofm/models/releases/download/4xNomos2_hq_drct-l/4xNomos2_hq_drct-l.pth", 
                                        "https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_drct-l",
"""Phhofm: An drct-l 4x upscaling model, similiar to the 4xNomos2_hq_atd, 4xNomos2_hq_dat2 and 4xNomos2_hq_mosr models, trained and for usage on non-degraded input to give good quality output.
"""],

    # ATD
    "4xNomos2_hq_atd.pth"             : ["https://github.com/Phhofm/models/releases/download/4xNomos2_hq_atd/4xNomos2_hq_atd.pth", 
                                         "https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_atd",
"""Phhofm: An atd 4x upscaling model, similiar to the 4xNomos2_hq_dat2 or 4xNomos2_hq_mosr models, trained and for usage on non-degraded input to give good quality output.
"""],

    # MoSR
    "4xNomos2_hq_mosr.pth"             : ["https://github.com/Phhofm/models/releases/download/4xNomos2_hq_mosr/4xNomos2_hq_mosr.pth", 
                                         "https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_mosr",
"""Phhofm: A 4x MoSR upscaling model, meant for non-degraded input, since this model was trained on non-degraded input to give good quality output.
"""],
    
    "2x-AnimeSharpV2_MoSR_Sharp.pth"             : ["https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV2_Set/2x-AnimeSharpV2_MoSR_Sharp.pth", 
                                         "https://github.com/Kim2091/Kim2091-Models/releases/tag/2x-AnimeSharpV2_Set",
"""Kim2091: This is my first anime model in years. Hopefully you guys can find a good use-case for it.
MoSR (Lower quality, faster), Sharp: For heavily degraded sources. Sharp models have issues depth of field but are best at removing artifacts
"""],
    
    "2x-AnimeSharpV2_MoSR_Soft.pth"             : ["https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV2_Set/2x-AnimeSharpV2_MoSR_Soft.pth", 
                                         "https://github.com/Kim2091/Kim2091-Models/releases/tag/2x-AnimeSharpV2_Set",
"""Kim2091: This is my first anime model in years. Hopefully you guys can find a good use-case for it.
MoSR (Lower quality, faster), Soft: For cleaner sources. Soft models preserve depth of field but may not remove other artifacts as well
"""],

    # SRFormer
    "4xNomos8kSCSRFormer.pth"             : ["https://github.com/Phhofm/models/releases/download/4xNomos8kSCSRFormer/4xNomos8kSCSRFormer.pth", 
                                             "https://github.com/Phhofm/models/releases/tag/4xNomos8kSCSRFormer",
"""Phhofm: 4x photo upscaler with otf jpg compression and blur, trained on musl's Nomos8k_sfw dataset for realisic sr.
"""],

#     "4xFrankendata_FullDegradation_g_460000.pth" : ["https://drive.google.com/uc?export=download&confirm=1&id=1PZrj-8ofxhORv_OgTVSoRt3dYi-BtiDj", 
#                                                     "https://openmodeldb.info/models/4x-Frankendata-FullDegradation-SRFormer",
# """Description: 4x realistic upscaler that may also work for general purpose usage. 
# It was trained with OTF random degradation with a very low to very high range of degradations, including blur, noise, and compression. 
# Trained with the same Frankendata dataset that I used for the pretrain model.
# """],

#     "FrankendataPretrainer_SRFormer400K_g.pth" : ["https://drive.google.com/uc?export=download&confirm=1&id=1SaKvpYYIm2Vj2m9GifUMlNCbmkE6JZmr", 
#                                                     "https://openmodeldb.info/models/4x-FrankendataPretainer-SRFormer",
# """Description: 4x realistic upscaler that may also work for general purpose usage. 
# It was trained with OTF random degradation with a very low to very high range of degradations, including blur, noise, and compression. 
# Trained with the same Frankendata dataset that I used for the pretrain model.
# """],

#     "1xFrankenfixer_SRFormerLight_g.pth" : ["https://drive.google.com/uc?export=download&confirm=1&id=1UJ0iyFn4IGNhPIgNgrQrBxYsdDloFc9I", 
#                                                   "https://openmodeldb.info/models/1x-Frankenfixer-SRFormerLight",
# """A 1x model designed to reduce artifacts and restore detail to images upscaled by 4xFrankendata_FullDegradation_SRFormer. It could possibly work with other upscaling models too.
# """],
}

example_list = ["images/a01.jpg", "images/a02.jpg", "images/a03.jpg", "images/a04.jpg", "images/bus.jpg", "images/zidane.jpg", 
                "images/b01.jpg", "images/b02.jpg", "images/b03.jpg", "images/b04.jpg", "images/b05.jpg", "images/b06.jpg", 
                "images/b07.jpg", "images/b08.jpg", "images/b09.jpg", "images/b10.jpg", "images/b11.jpg", "images/c01.jpg",  
                "images/c02.jpg", "images/c03.jpg", "images/c04.jpg", "images/c05.jpg", "images/c06.jpg", "images/c07.jpg", 
                "images/c08.jpg", "images/c09.jpg", "images/c10.jpg"]

def get_model_type(model_name):
    # Define model type mappings based on key parts of the model names
    model_type = "other"
    if any(value in model_name.lower() for value in ("4x-animesharp.pth", "sudo-realesrgan")):
        model_type = "ESRGAN"
    elif "srformer" in model_name.lower():
        model_type = "SRFormer"
    elif ("realplksr" in model_name.lower() and "dysample" in model_name.lower()) or "rplksrd" in model_name.lower():
        model_type = "RealPLKSR_dysample"
    elif any(value in model_name.lower() for value in ("realplksr", "rplksr", "realplskr")):
        model_type = "RealPLKSR"
    elif any(value in model_name.lower() for value in ("realesrgan", "realesrnet")):
        model_type = "RRDB"
    elif any(value in model_name.lower() for value in ("realesr", "exposurecorrection", "parimgcompact", "lsdircompact")):
        model_type = "SRVGG"
    elif "esrgan" in model_name.lower():
        model_type = "ESRGAN"
    elif "dat" in model_name.lower():
        model_type = "DAT"
    elif "hat" in model_name.lower():
        model_type = "HAT"
    elif "drct" in model_name.lower():
        model_type = "DRCT"
    elif "atd" in model_name.lower():
        model_type = "ATD"
    elif "mosr" in model_name.lower():
        model_type = "MoSR"
    return f"{model_type}, {model_name}"

typed_upscale_models = {get_model_type(key): value[0] for key, value in upscale_models.items()}


class Upscale:
    def inference(self, img, face_restoration, upscale_model, scale: float, face_detection, face_detection_threshold: any, face_detection_only_center: bool, outputWithModelName: bool):
        print(img)
        print(face_restoration, upscale_model, scale)
        try:
            self.scale = scale
            self.img_name = os.path.basename(str(img))
            self.basename, self.extension = os.path.splitext(self.img_name)
            
            img = cv2.imdecode(np.fromfile(img, np.uint8), cv2.IMREAD_UNCHANGED) # numpy.ndarray
            
            self.img_mode = "RGBA" if len(img.shape) == 3 and img.shape[2] == 4 else None
            if len(img.shape) == 2:  # for gray inputs
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            self.h_input, self.w_input = img.shape[0:2]

            if face_restoration:
                download_from_url(face_models[face_restoration][0], face_restoration, os.path.join("weights", "face"))
                
            modelInUse = ""
            upscale_type = None
            if upscale_model:
                upscale_type, upscale_model = upscale_model.split(", ", 1)
                download_from_url(upscale_models[upscale_model][0], upscale_model, os.path.join("weights", "upscale"))
                modelInUse = f"_{os.path.splitext(upscale_model)[0]}"
            
            self.netscale = 1 if any(sub in upscale_model.lower() for sub in ("x1", "1x")) else (2 if any(sub in upscale_model.lower() for sub in ("x2", "2x")) else 4)
            model = None
            is_auto_split_upscale = True
            half = True if torch.cuda.is_available() else False
            if upscale_type:
                # The values of the following hyperparameters are based on the research findings of the Spandrel project.
                # https://github.com/chaiNNer-org/spandrel/tree/main/libs/spandrel/spandrel/architectures
                from basicsr.archs.rrdbnet_arch import RRDBNet
                loadnet = torch.load(os.path.join("weights", "upscale", upscale_model), map_location=torch.device('cpu'), weights_only=True)
                if 'params_ema' in loadnet or 'params' in loadnet:
                    loadnet = loadnet['params_ema'] if 'params_ema' in loadnet else loadnet['params']
                # for key in loadnet_origin.keys():
                #     print(f"{key}, {loadnet_origin[key].shape}")
                if upscale_type == "SRVGG":
                    from basicsr.archs.srvgg_arch import SRVGGNetCompact
                    body_max_num = self.find_max_numbers(loadnet, "body")
                    num_feat     = loadnet["body.0.weight"].shape[0]
                    num_in_ch    = loadnet["body.0.weight"].shape[1]
                    num_conv     = body_max_num // 2 - 1
                    model        = SRVGGNetCompact(num_in_ch=num_in_ch, num_out_ch=3, num_feat=num_feat, num_conv=num_conv, upscale=self.netscale, act_type='prelu')
                elif upscale_type == "RRDB" or upscale_type == "ESRGAN":
                    if upscale_type == "RRDB":
                        num_block = self.find_max_numbers(loadnet, "body") + 1
                        num_feat  = loadnet["conv_first.weight"].shape[0]
                    else:
                        num_block = self.find_max_numbers(loadnet, "model.1.sub")
                        num_feat  = loadnet["model.0.weight"].shape[0]
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_block=num_block, num_grow_ch=32, scale=self.netscale, is_real_esrgan=upscale_type == "RRDB")
                elif upscale_type == "DAT":
                    from basicsr.archs.dat_arch import DAT
                    half = False

                    in_chans   = loadnet["conv_first.weight"].shape[1]
                    embed_dim  = loadnet["conv_first.weight"].shape[0]
                    num_layers = self.find_max_numbers(loadnet, "layers") + 1
                    depth      = [6] * num_layers
                    num_heads  = [6] * num_layers
                    for i in range(num_layers):
                        depth[i] = self.find_max_numbers(loadnet, f"layers.{i}.blocks") + 1
                        num_heads[i] = loadnet[f"layers.{i}.blocks.1.attn.temperature"].shape[0] if depth[i] >= 2 else \
                                       loadnet[f"layers.{i}.blocks.0.attn.attns.0.pos.pos3.2.weight"].shape[0] * 2

                    upsampler        = "pixelshuffle" if "conv_last.weight" in loadnet else "pixelshuffledirect"
                    resi_connection  = "1conv" if "conv_after_body.weight" in loadnet else "3conv"
                    qkv_bias         = "layers.0.blocks.0.attn.qkv.bias" in loadnet
                    expansion_factor = float(loadnet["layers.0.blocks.0.ffn.fc1.weight"].shape[0] / embed_dim)

                    img_size = 64
                    if "layers.0.blocks.2.attn.attn_mask_0" in loadnet:
                        attn_mask_0_x, attn_mask_0_y, _attn_mask_0_z = loadnet["layers.0.blocks.2.attn.attn_mask_0"].shape
                        img_size = int(math.sqrt(attn_mask_0_x * attn_mask_0_y))

                    split_size = [2, 4]
                    if "layers.0.blocks.0.attn.attns.0.rpe_biases" in loadnet:
                        split_sizes = loadnet["layers.0.blocks.0.attn.attns.0.rpe_biases"][-1] + 1
                        split_size = [int(x) for x in split_sizes]

                    model = DAT(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, split_size=split_size, depth=depth, num_heads=num_heads, expansion_factor=expansion_factor, 
                                qkv_bias=qkv_bias, resi_connection=resi_connection, upsampler=upsampler, upscale=self.netscale)
                elif upscale_type == "HAT":
                    half = False
                    from basicsr.archs.hat_arch import HAT
                    in_chans = loadnet["conv_first.weight"].shape[1]
                    embed_dim = loadnet["conv_first.weight"].shape[0]
                    window_size = int(math.sqrt(loadnet["relative_position_index_SA"].shape[0]))
                    num_layers = self.find_max_numbers(loadnet, "layers") + 1
                    depths      = [6] * num_layers
                    num_heads   = [6] * num_layers
                    for i in range(num_layers):
                        depths[i] = self.find_max_numbers(loadnet, f"layers.{i}.residual_group.blocks") + 1
                        num_heads[i] = loadnet[f"layers.{i}.residual_group.overlap_attn.relative_position_bias_table"].shape[1]
                    resi_connection = "1conv" if "conv_after_body.weight" in loadnet else "identity"

                    qkv_bias = "layers.0.residual_group.blocks.0.attn.qkv.bias" in loadnet
                    patch_norm = "patch_embed.norm.weight" in loadnet
                    ape = "absolute_pos_embed" in loadnet

                    mlp_hidden_dim = int(loadnet["layers.0.residual_group.blocks.0.mlp.fc1.weight"].shape[0])
                    mlp_ratio = mlp_hidden_dim / embed_dim
                    upsampler = "pixelshuffle"

                    if "hat-l" in upscale_model.lower():
                        compress_ratio = 3
                        squeeze_factor = 30
                    elif "hat-s" in upscale_model.lower():
                        compress_ratio = 24
                        squeeze_factor = 24
                    model = HAT(img_size=64, patch_size=1, in_chans=in_chans, embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size, compress_ratio=compress_ratio,
                                squeeze_factor=squeeze_factor, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, ape=ape, patch_norm=patch_norm,
                                upsampler=upsampler, resi_connection=resi_connection, upscale=self.netscale,)
                elif "RealPLKSR" in upscale_type:
                    from basicsr.archs.realplksr_arch import realplksr
                    half = False if "RealPLSKR" in upscale_model else half
                    use_ea       = "feats.1.attn.f.0.weight" in loadnet
                    dim          = loadnet["feats.0.weight"].shape[0]
                    num_feats    = self.find_max_numbers(loadnet, "feats") + 1
                    n_blocks     = num_feats - 3
                    kernel_size  = loadnet["feats.1.lk.conv.weight"].shape[2]
                    split_ratio  = loadnet["feats.1.lk.conv.weight"].shape[0] / dim
                    use_dysample = "to_img.init_pos" in loadnet

                    model = realplksr(upscaling_factor=self.netscale, dim=dim, n_blocks=n_blocks, kernel_size=kernel_size, split_ratio=split_ratio, use_ea=use_ea, dysample=use_dysample)
                elif upscale_type == "DRCT":
                    half = False
                    from basicsr.archs.DRCT_arch import DRCT

                    in_chans    = loadnet["conv_first.weight"].shape[1]
                    embed_dim   = loadnet["conv_first.weight"].shape[0]
                    num_layers  = self.find_max_numbers(loadnet, "layers") + 1
                    depths      = (6,) * num_layers
                    num_heads   = []
                    for i in range(num_layers):
                        num_heads.append(loadnet[f"layers.{i}.swin1.attn.relative_position_bias_table"].shape[1])

                    mlp_ratio       = loadnet["layers.0.swin1.mlp.fc1.weight"].shape[0] / embed_dim
                    window_square   = loadnet["layers.0.swin1.attn.relative_position_bias_table"].shape[0]
                    window_size     = (math.isqrt(window_square) + 1) // 2
                    upsampler       = "pixelshuffle" if "conv_last.weight" in loadnet else ""
                    resi_connection = "1conv" if "conv_after_body.weight" in loadnet else ""
                    qkv_bias        = "layers.0.swin1.attn.qkv.bias" in loadnet
                    gc_adjust1      = loadnet["layers.0.adjust1.weight"].shape[0]
                    patch_norm      = "patch_embed.norm.weight" in loadnet
                    ape             = "absolute_pos_embed" in loadnet

                    model = DRCT(in_chans=in_chans,  img_size= 64, window_size=window_size, compress_ratio=3,squeeze_factor=30,
                        conv_scale= 0.01, overlap_ratio= 0.5, img_range= 1., depths=depths, embed_dim=embed_dim, num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, ape=ape, patch_norm=patch_norm, use_checkpoint=False,
                        upscale=self.netscale, upsampler=upsampler, resi_connection=resi_connection, gc =gc_adjust1,)
                elif upscale_type == "ATD":
                    half = False
                    from basicsr.archs.atd_arch import ATD
                    in_chans    = loadnet["conv_first.weight"].shape[1]
                    embed_dim   = loadnet["conv_first.weight"].shape[0]
                    window_size = math.isqrt(loadnet["relative_position_index_SA"].shape[0])
                    num_layers  = self.find_max_numbers(loadnet, "layers") + 1
                    depths      = [6] * num_layers
                    num_heads   = [6] * num_layers
                    for i in range(num_layers):
                        depths[i] = self.find_max_numbers(loadnet, f"layers.{i}.residual_group.layers") + 1
                        num_heads[i] = loadnet[f"layers.{i}.residual_group.layers.0.attn_win.relative_position_bias_table"].shape[1]
                    num_tokens          = loadnet["layers.0.residual_group.layers.0.attn_atd.scale"].shape[0]
                    reducted_dim        = loadnet["layers.0.residual_group.layers.0.attn_atd.wq.weight"].shape[0]
                    convffn_kernel_size = loadnet["layers.0.residual_group.layers.0.convffn.dwconv.depthwise_conv.0.weight"].shape[2]
                    mlp_ratio           = (loadnet["layers.0.residual_group.layers.0.convffn.fc1.weight"].shape[0] / embed_dim)
                    qkv_bias            = "layers.0.residual_group.layers.0.wqkv.bias" in loadnet
                    ape                 = "absolute_pos_embed" in loadnet
                    patch_norm          = "patch_embed.norm.weight" in loadnet
                    resi_connection     = "1conv" if "layers.0.conv.weight" in loadnet else "3conv"

                    if "conv_up1.weight" in loadnet:
                        upsampler = "nearest+conv"
                    elif "conv_before_upsample.0.weight" in loadnet:
                        upsampler = "pixelshuffle"
                    elif "conv_last.weight" in loadnet:
                        upsampler = ""
                    else:
                        upsampler = "pixelshuffledirect"

                    is_light = upsampler == "pixelshuffledirect" and embed_dim == 48
                    category_size = 128 if is_light else 256

                    model = ATD(in_chans=in_chans, embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size, category_size=category_size,
                                num_tokens=num_tokens, reducted_dim=reducted_dim, convffn_kernel_size=convffn_kernel_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, ape=ape,
                                patch_norm=patch_norm, use_checkpoint=False, upscale=self.netscale, upsampler=upsampler, resi_connection='1conv',)
                elif upscale_type == "MoSR":
                    from basicsr.archs.mosr_arch import mosr
                    n_block         = self.find_max_numbers(loadnet, "gblocks") - 5
                    in_ch           = loadnet["gblocks.0.weight"].shape[1]
                    out_ch          = loadnet["upsampler.end_conv.weight"].shape[0] if "upsampler.init_pos" in loadnet else in_ch
                    dim             = loadnet["gblocks.0.weight"].shape[0]
                    expansion_ratio = (loadnet["gblocks.1.fc1.weight"].shape[0] / loadnet["gblocks.1.fc1.weight"].shape[1]) / 2
                    conv_ratio      = loadnet["gblocks.1.conv.weight"].shape[0] / dim
                    kernel_size     = loadnet["gblocks.1.conv.weight"].shape[2]
                    upsampler       = "dys" if "upsampler.init_pos" in loadnet else ("gps" if "upsampler.in_to_k.weight" in loadnet else "ps")

                    model = mosr(in_ch = in_ch, out_ch = out_ch, upscale = self.netscale, n_block = n_block, dim = dim,
                                upsampler = upsampler, kernel_size = kernel_size, expansion_ratio = expansion_ratio, conv_ratio = conv_ratio,)
                elif upscale_type == "SRFormer":
                    half = False
                    from basicsr.archs.srformer_arch import SRFormer
                    in_chans   = loadnet["conv_first.weight"].shape[1]
                    embed_dim  = loadnet["conv_first.weight"].shape[0]
                    ape        = "absolute_pos_embed" in loadnet
                    patch_norm = "patch_embed.norm.weight" in loadnet
                    qkv_bias   = "layers.0.residual_group.blocks.0.attn.q.bias" in loadnet
                    mlp_ratio  = float(loadnet["layers.0.residual_group.blocks.0.mlp.fc1.weight"].shape[0] / embed_dim)

                    num_layers = self.find_max_numbers(loadnet, "layers") + 1
                    depths     = [6] * num_layers
                    num_heads  = [6] * num_layers
                    for i in range(num_layers):
                        depths[i] = self.find_max_numbers(loadnet, f"layers.{i}.residual_group.blocks") + 1
                        num_heads[i] = loadnet[f"layers.{i}.residual_group.blocks.0.attn.relative_position_bias_table"].shape[1]

                    if "conv_hr.weight" in loadnet:
                        upsampler = "nearest+conv"
                    elif "conv_before_upsample.0.weight" in loadnet:
                        upsampler = "pixelshuffle"
                    elif "upsample.0.weight" in loadnet:
                        upsampler = "pixelshuffledirect"
                    resi_connection = "1conv" if "conv_after_body.weight" in loadnet else "3conv"

                    window_size = int(math.sqrt(loadnet["layers.0.residual_group.blocks.0.attn.relative_position_bias_table"].shape[0])) + 1

                    model = SRFormer(img_size=64, in_chans=in_chans, embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio, 
                                 qkv_bias=qkv_bias, qk_scale=None, ape=ape, patch_norm=patch_norm, upscale=self.netscale, upsampler=upsampler, resi_connection=resi_connection,)

            self.upsampler = None
            if model:
                self.upsampler = RealESRGANer(scale=self.netscale, model_path=os.path.join("weights", "upscale", upscale_model), model=model, tile=0, tile_pad=10, pre_pad=0, half=half)
            elif upscale_model:
                self.upsampler = None
                import PIL
                from image_gen_aux import UpscaleWithModel
                class UpscaleWithModel_Gfpgan(UpscaleWithModel):
                    def cv2pil(self, image):
                        ''' OpenCV type -> PIL type
                        https://qiita.com/derodero24/items/f22c22b22451609908ee
                        '''
                        new_image = image.copy()
                        if new_image.ndim == 2:  # Grayscale
                            pass
                        elif new_image.shape[2] == 3:  # Color
                            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
                        elif new_image.shape[2] == 4:  # Transparency
                            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
                        new_image = PIL.Image.fromarray(new_image)
                        return new_image

                    def pil2cv(self, image):
                        ''' PIL type -> OpenCV type
                        https://qiita.com/derodero24/items/f22c22b22451609908ee
                        '''
                        new_image = np.array(image, dtype=np.uint8)
                        if new_image.ndim == 2:  # Grayscale
                            pass
                        elif new_image.shape[2] == 3:  # Color
                            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                        elif new_image.shape[2] == 4:  # Transparency
                            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
                        return new_image

                    def enhance(self_, img, outscale=None):
                        # img: numpy
                        h_input, w_input = img.shape[0:2]
                        pil_img = self.cv2pil(img)
                        pil_img = self_.__call__(pil_img)
                        cv_image = self.pil2cv(pil_img)
                        if outscale is not None and outscale != float(self.netscale):
                            interpolation = cv2.INTER_AREA if outscale < float(self.netscale) else cv2.INTER_LANCZOS4
                            cv_image = cv2.resize(
                                cv_image, (
                                    int(w_input * outscale),
                                    int(h_input * outscale),
                                ), interpolation=interpolation)
                        return cv_image, None

                device = "cuda" if torch.cuda.is_available() else "cpu"
                upscaler = UpscaleWithModel.from_pretrained(os.path.join("weights", "upscale", upscale_model)).to(device)
                upscaler.__class__ = UpscaleWithModel_Gfpgan
                self.upsampler = upscaler
            self.face_enhancer = None

            resolution = 512
            if face_restoration:
                modelInUse = f"_{os.path.splitext(face_restoration)[0]}" + modelInUse
                from gfpgan.utils import GFPGANer
                model_rootpath = os.path.join("weights", "face")
                model_path = os.path.join(model_rootpath, face_restoration)
                channel_multiplier = None

                if face_restoration and face_restoration.startswith("GFPGANv1."):
                    arch = "clean"
                    channel_multiplier = 2
                elif face_restoration and face_restoration.startswith("RestoreFormer"):
                    arch = "RestoreFormer++" if face_restoration.startswith("RestoreFormer++") else "RestoreFormer"
                elif face_restoration == 'CodeFormer.pth':
                    arch = "CodeFormer"
                elif face_restoration.startswith("GPEN-BFR-"):
                    arch = "GPEN"
                    channel_multiplier = 2
                    if "1024" in face_restoration:
                        arch = "GPEN-1024"
                        resolution = 1024
                    elif "2048" in face_restoration:
                        arch = "GPEN-2048"
                        resolution = 2048

                self.face_enhancer = GFPGANer(model_path=model_path, upscale=self.scale, arch=arch, channel_multiplier=channel_multiplier, model_rootpath=model_rootpath, det_model=face_detection, resolution=resolution)

            files = []
            if not outputWithModelName:
                modelInUse = ""

            try:
                bg_upsample_img = None
                if self.upsampler and hasattr(self.upsampler, "enhance"):
                    from utils.dataops import auto_split_upscale
                    bg_upsample_img, _ = auto_split_upscale(img, self.upsampler.enhance, self.scale) if is_auto_split_upscale else self.upsampler.enhance(img, outscale=self.scale)

                if self.face_enhancer:
                    cropped_faces, restored_aligned, bg_upsample_img = self.face_enhancer.enhance(img, has_aligned=False, only_center_face=face_detection_only_center, paste_back=True, bg_upsample_img=bg_upsample_img, eye_dist_threshold=face_detection_threshold)
                    # save faces
                    if cropped_faces and restored_aligned:
                        for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_aligned)):
                            # save cropped face
                            save_crop_path = f"output/{self.basename}{idx:02d}_cropped_faces{modelInUse}.png"
                            self.imwriteUTF8(save_crop_path, cropped_face)
                            # save restored face
                            save_restore_path = f"output/{self.basename}{idx:02d}_restored_faces{modelInUse}.png"
                            self.imwriteUTF8(save_restore_path, restored_face)
                            # save comparison image
                            save_cmp_path = f"output/{self.basename}{idx:02d}_cmp{modelInUse}.png"
                            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                            self.imwriteUTF8(save_cmp_path, cmp_img)
        
                            files.append(save_crop_path)
                            files.append(save_restore_path)
                            files.append(save_cmp_path)

                restored_img = bg_upsample_img
            except RuntimeError as error:
                print(traceback.format_exc())
                print('Error', error)
            finally:
                if self.face_enhancer:
                    self.face_enhancer._cleanup()
                else:
                    # Free GPU memory and clean up resources
                    torch.cuda.empty_cache()
                    gc.collect()

            if not self.extension:
                self.extension = ".png" if self.img_mode == "RGBA" else ".jpg" # RGBA images should be saved in png format
            save_path = f"output/{self.basename}{modelInUse}{self.extension}"
            self.imwriteUTF8(save_path, restored_img)

            restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
            files.append(save_path)
            return files, files
        except Exception as error:
            print(traceback.format_exc())
            print("global exception", error)
            return None, None

    def find_max_numbers(self, state_dict, findkeys):
        if isinstance(findkeys, str):
            findkeys = [findkeys]
        max_values = defaultdict(lambda: None)
        patterns = {findkey: re.compile(rf"^{re.escape(findkey)}\.(\d+)\.") for findkey in findkeys}
    
        for key in state_dict:
            for findkey, pattern in patterns.items():
                if match := pattern.match(key):  
                    num = int(match.group(1))
                    max_values[findkey] = max(num, max_values[findkey] if max_values[findkey] is not None else num)

        return tuple(max_values[findkey] for findkey in findkeys) if len(findkeys) > 1 else max_values[findkeys[0]]

    def imwriteUTF8(self, save_path, image): # `cv2.imwrite` does not support writing files to UTF-8 file paths.
        img_name = os.path.basename(save_path)
        _, extension = os.path.splitext(img_name)
        is_success, im_buf_arr = cv2.imencode(extension, image)
        if (is_success): im_buf_arr.tofile(save_path)


def main():
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.975, device='cuda:0')
        # set torch options to avoid get black image for RTX16xx card
        # https://github.com/CompVis/stable-diffusion/issues/69#issuecomment-1260722801
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    # Ensure the target directory exists
    os.makedirs('output', exist_ok=True)

    title = "Image Upscaling & Restoration using GFPGAN / RestoreFormerPlusPlus / CodeFormer / GPEN Algorithm"
    description = r"""
    <a href='https://github.com/TencentARC/GFPGAN' target='_blank'><b>GFPGAN: Towards Real-World Blind Face Restoration and Upscalling of the image with a Generative Facial Prior</b></a>. <br>
    <a href='https://github.com/wzhouxiff/RestoreFormerPlusPlus' target='_blank'><b>RestoreFormer++: Towards Real-World Blind Face Restoration from Undegraded Key-Value Pairs</b></a>. <br>
    <a href='https://github.com/sczhou/CodeFormer' target='_blank'><b>CodeFormer: Towards Robust Blind Face Restoration with Codebook Lookup Transformer (NeurIPS 2022)</b></a>. <br>
    <a href='https://github.com/yangxy/GPEN' target='_blank'><b>GPEN: GAN Prior Embedded Network for Blind Face Restoration in the Wild</b></a>. <br>
    <br>
    Practically, the aforementioned algorithm is used to restore your **old photos** or improve **AI-generated faces**.<br>
    To use it, simply just upload the concerned image.<br>
    """

    upscale = Upscale()
    
    rows = []
    tmptype = None
    upscale_model_tables = []
    for key, _ in typed_upscale_models.items():
        upscale_type, upscale_model = key.split(", ", 1)
        if tmptype and tmptype != upscale_type:#RRDB ESRGAN
            speed = "Fast" if tmptype == "SRVGG" else ("Slow" if any(value == tmptype for value in ("DAT", "HAT", "DRCT", "ATD", "SRFormer")) else "Normal")
            upscale_model_header = f"| Upscale Model | Info, Type: {tmptype}, Model execution speed: {speed} | Download URL |\n|------------|------|--------------|"
            upscale_model_tables.append(upscale_model_header + "\n" + "\n".join(rows))
            rows.clear()
        tmptype = upscale_type
        value = upscale_models[upscale_model]
        row = f"| [{upscale_model}]({value[1]}) | " + value[2].replace("\n", "<br>") + " | [download]({value[0]}) |"
        rows.append(row)
    speed = "Fast" if tmptype == "SRVGG" else ("Slow" if any(value == tmptype for value in ("DAT", "HAT", "DRCT", "ATD", "SRFormer")) else "Normal")
    upscale_model_header = f"| Upscale Model Name | Info, Type: {tmptype}, Model execution speed: {speed} | Download URL |\n|------------|------|--------------|"
    upscale_model_tables.append(upscale_model_header + "\n" + "\n".join(rows))

    with gr.Blocks(title = title) as demo:
        gr.Markdown(value=f"<h1 style=\"text-align:center;\">{title}</h1><br>{description}")
        with gr.Row():
            with gr.Column(variant         ="panel"):
                input_image                = gr.Image(type="filepath", label="Input", format="png")
                face_model                 = gr.Dropdown(list(face_models.keys())+[None], type="value", value='GFPGANv1.4.pth', label='Face Restoration version', info="Face Restoration and RealESR can be freely combined in different ways, or one can be set to \"None\" to use only the other model. Face Restoration is primarily used for face restoration in real-life images, while RealESR serves as a background restoration model.")
                upscale_model              = gr.Dropdown(list(typed_upscale_models.keys())+[None], type="value", value='SRVGG, realesr-general-x4v3.pth', label='UpScale version')
                upscale_scale              = gr.Number(label="Rescaling factor", value=4)
                face_detection             = gr.Dropdown(["retinaface_resnet50", "YOLOv5l", "YOLOv5n"], type="value", value="retinaface_resnet50", label="Face Detection type")
                face_detection_threshold   = gr.Number(label="Face eye dist threshold", value=10, info="A threshold to filter out faces with too small an eye distance (e.g., side faces).")
                face_detection_only_center = gr.Checkbox(value=False, label="Face detection only center", info="If set to True, only the face closest to the center of the image will be kept.")
                with_model_name  = gr.Checkbox(label="Output image files name with model name", value=True)
                with gr.Row():
                    submit = gr.Button(value="Submit", variant="primary", size="lg")
                    clear = gr.ClearButton(
                        components=[
                            input_image,
                            face_model,
                            upscale_model,
                            upscale_scale,
                            face_detection,
                            face_detection_threshold,
                            face_detection_only_center,
                            with_model_name,
                        ], variant="secondary", size="lg",)
            with gr.Column(variant="panel"):
                gallerys = gr.Gallery(type="filepath", label="Output (The whole image)", format="png")
                outputs = gr.File(label="Download the output image")
        with gr.Row(variant="panel"):
            # Generate output array
            output_arr = []
            for file_name in example_list:
                output_arr.append([file_name,])
            gr.Examples(output_arr, inputs=[input_image,], examples_per_page=20)
        with gr.Row(variant="panel"):
            # Convert to Markdown table
            header = "| Face Model Name | Info | Download URL |\n|------------|------|--------------|"
            rows = [
                f"| [{key}]({value[1]}) | " + value[2].replace("\n", "<br>") + f" | [download]({value[0]}) |"
                for key, value in face_models.items()
            ]
            markdown_table = header + "\n" + "\n".join(rows)
            gr.Markdown(value=markdown_table)

        for table in upscale_model_tables:
            with gr.Row(variant="panel"):
                gr.Markdown(value=table)

        submit.click(
            upscale.inference, 
            inputs=[
                input_image,
                face_model,
                upscale_model,
                upscale_scale,
                face_detection,
                face_detection_threshold,
                face_detection_only_center,
                with_model_name,
            ],
            outputs=[gallerys, outputs],
        )
    
    demo.queue(default_concurrency_limit=1)
    demo.launch(inbrowser=True)


if __name__ == "__main__":
    main()
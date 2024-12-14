import os
import gc
import cv2
import numpy as np
import gradio as gr
import torch
import traceback
from facexlib.utils.misc import download_from_url
from realesrgan.utils import RealESRGANer


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

    # DATNet
    "4xNomos8kDAT.pth"                     : ["https://github.com/Phhofm/models/releases/download/4xNomos8kDAT/4xNomos8kDAT.pth",
                                             "https://openmodeldb.info/models/4x-Nomos8kDAT", 
"""A 4x photo upscaler with otf jpg compression, blur and resize, trained on musl's Nomos8k_sfw dataset for realisic sr, this time based on the DAT arch, as a finetune on the official 4x DAT model."""],

    "4x-DWTP-DS-dat2-v3.pth"               : ["https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/4x-DWTP-DS-dat2-v3.pth",
                                             "https://openmodeldb.info/models/4x-DWTP-DS-dat2-v3", 
"""DAT descreenton model, designed to reduce discrepancies on tiles due to too much loss of the first version, while getting rid of the removal of paper texture"""],

    "4xBHI_dat2_real.pth"                  : ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_real/4xBHI_dat2_real.pth",
                                             "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_real", 
"""Purpose: 4x upscaling images. Handles realistic noise, some realistic blur, and webp and jpg (re)compression.
Description: 4x dat2 upscaling model for web and realistic images. It handles realistic noise, some realistic blur, and webp and jpg (re)compression. Trained on my BHI dataset (390'035 training tiles) with degraded LR subset."""],

    "4xBHI_dat2_otf.pth"                   : ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_otf/4xBHI_dat2_otf.pth",
                                             "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_otf", 
"""Purpose: 4x upscaling images, handles noise and jpg compression
Description: 4x dat2 upscaling model, trained with the real-esrgan otf pipeline on my bhi dataset. Handles noise and compression."""],

    "4xBHI_dat2_multiblur.pth"             : ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_multiblurjpg/4xBHI_dat2_multiblur.pth",
                                             "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_multiblurjpg", 
"""Purpose: 4x upscaling images, handles jpg compression
Description: 4x dat2 upscaling model, trained with down_up,linear, cubic_mitchell, lanczos, gauss and box scaling algos, some average, gaussian and anisotropic blurs and jpg compression. Trained on my BHI sisr dataset."""],

    "4xBHI_dat2_multiblurjpg.pth"          : ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_multiblurjpg/4xBHI_dat2_multiblurjpg.pth",
                                             "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_multiblurjpg", 
"""Purpose: 4x upscaling images, handles jpg compression
Description: 4x dat2 upscaling model, trained with down_up,linear, cubic_mitchell, lanczos, gauss and box scaling algos, some average, gaussian and anisotropic blurs and jpg compression. Trained on my BHI sisr dataset."""],

    "4x_IllustrationJaNai_V1_DAT2_190k.pth": ["https://drive.google.com/uc?export=download&confirm=1&id=1qpioSqBkB_IkSBhEAewSSNFt6qgkBimP",
                                             "https://openmodeldb.info/models/4x-IllustrationJaNai-V1-DAT2", 
"""Purpose: Illustrations, digital art, manga covers
Model for color images including manga covers and color illustrations, digital art, visual novel art, artbooks, and more. 
DAT2 version is the highest quality version but also the slowest. See the ESRGAN version for faster performance."""],

    # HAT
    "4xNomos8kSCHAT-L.pth"  : ["https://github.com/Phhofm/models/releases/download/4xNomos8kSCHAT/4xNomos8kSCHAT-L.pth",
                              "https://openmodeldb.info/models/4x-Nomos8kSCHAT-L", 
"""4x photo upscaler with otf jpg compression and blur, trained on musl's Nomos8k_sfw dataset for realisic sr. Since this is a big model, upscaling might take a while."""],

    "4xNomos8kSCHAT-S.pth"  : ["https://github.com/Phhofm/models/releases/download/4xNomos8kSCHAT/4xNomos8kSCHAT-S.pth",
                              "https://openmodeldb.info/models/4x-Nomos8kSCHAT-S", 
"""4x photo upscaler with otf jpg compression and blur, trained on musl's Nomos8k_sfw dataset for realisic sr. HAT-S version/model."""],

    "4xNomos8kHAT-L_otf.pth": ["https://github.com/Phhofm/models/releases/download/4xNomos8kHAT-L_otf/4xNomos8kHAT-L_otf.pth",
                              "https://openmodeldb.info/models/4x-Nomos8kHAT-L-otf", 
"""4x photo upscaler trained with otf"""],

    # RealPLKSR_dysample
    "4xHFA2k_ludvae_realplksr_dysample.pth": ["https://github.com/Phhofm/models/releases/download/4xHFA2k_ludvae_realplksr_dysample/4xHFA2k_ludvae_realplksr_dysample.pth",
                                             "https://openmodeldb.info/models/4x-HFA2k-ludvae-realplksr-dysample", 
"""A Dysample RealPLKSR 4x upscaling model for anime single-image resolution."""],

    "4xArtFaces_realplksr_dysample.pth"    : ["https://github.com/Phhofm/models/releases/download/4xArtFaces_realplksr_dysample/4xArtFaces_realplksr_dysample.pth",
                                             "https://openmodeldb.info/models/4x-ArtFaces-realplksr-dysample", 
"""A Dysample RealPLKSR 4x upscaling model for art / painted faces."""],

    "4x-PBRify_RPLKSRd_V3.pth"             : ["https://github.com/Kim2091/Kim2091-Models/releases/download/4x-PBRify_RPLKSRd_V3/4x-PBRify_RPLKSRd_V3.pth", "https://openmodeldb.info/models/4x-PBRify-RPLKSRd-V3", 
"""This model is roughly 8x faster than the current DAT2 model, while being higher quality. It produces far more natural detail, resolves lines and edges more smoothly, and cleans up compression artifacts better."""],

    "4xNomos2_realplksr_dysample.pth"      : ["https://github.com/Phhofm/models/releases/download/4xNomos2_realplksr_dysample/4xNomos2_realplksr_dysample.pth",
                                             "https://openmodeldb.info/models/4x-Nomos2-realplksr-dysample", 
"""Description: A Dysample RealPLKSR 4x upscaling model that was trained with / handles jpg compression down to 70 on the Nomosv2 dataset, preserves DoF.
This model affects / saturate colors, which can be counteracted a bit by using wavelet color fix, as used in these examples."""],

    # RealPLKSR
    "2x-AnimeSharpV2_RPLKSR_Sharp.pth": ["https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV2_Set/2x-AnimeSharpV2_RPLKSR_Sharp.pth",
                                        "https://openmodeldb.info/models/2x-AnimeSharpV2-RPLKSR-Sharp", 
"""Kim2091: This is my first anime model in years. Hopefully you guys can find a good use-case for it.
RealPLKSR (Higher quality, slower) Sharp: For heavily degraded sources. Sharp models have issues depth of field but are best at removing artifacts
"""],

    "2x-AnimeSharpV2_RPLKSR_Soft.pth" : ["https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV2_Set/2x-AnimeSharpV2_RPLKSR_Soft.pth",
                                         "https://openmodeldb.info/models/2x-AnimeSharpV2-RPLKSR-Soft", 
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
"""4x RealPLKSR model for photography, trained with realistic noise, lens blur, jpg and webp re-compression."""],

#     "4xNomos2_hq_drct-l.pth"          : ["https://github.com/Phhofm/models/releases/download/4xNomos2_hq_drct-l/4xNomos2_hq_drct-l.pth", 
#                                         "https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_drct-l",
# """An drct-l 4x upscaling model, similiar to the 4xNomos2_hq_atd, 4xNomos2_hq_dat2 and 4xNomos2_hq_mosr models, trained and for usage on non-degraded input to give good quality output.
# """],

#     "4xNomos2_hq_atd.pth"             : ["https://github.com/Phhofm/models/releases/download/4xNomos2_hq_atd/4xNomos2_hq_atd.pth", 
#                                          "https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_atd",
# """An atd 4x upscaling model, similiar to the 4xNomos2_hq_dat2 or 4xNomos2_hq_mosr models, trained and for usage on non-degraded input to give good quality output.
# """]
}

example_list = ["images/a01.jpg", "images/a02.jpg", "images/a03.jpg", "images/a04.jpg", "images/bus.jpg", "images/zidane.jpg", 
                "images/b01.jpg", "images/b02.jpg", "images/b03.jpg", "images/b04.jpg", "images/b05.jpg", "images/b06.jpg", 
                "images/b07.jpg", "images/b08.jpg", "images/b09.jpg", "images/b10.jpg", "images/b11.jpg", "images/c01.jpg",  
                "images/c02.jpg", "images/c03.jpg", "images/c04.jpg", "images/c05.jpg", "images/c06.jpg", "images/c07.jpg", 
                "images/c08.jpg", "images/c09.jpg", "images/c10.jpg"]

def get_model_type(model_name):
    # Define model type mappings based on key parts of the model names
    model_type = "other"
    if any(value in model_name.lower() for value in ("realesrgan", "realesrnet")):
        model_type = "RRDB"
    elif "realesr" in model_name.lower() in model_name.lower():
        model_type = "SRVGG"
    elif "esrgan" in model_name.lower() or "4x-AnimeSharp.pth" == model_name:
        model_type = "ESRGAN"
    elif "dat" in model_name.lower():
        model_type = "DAT"
    elif "hat" in model_name.lower():
        model_type = "HAT"
    elif ("realplksr" in model_name.lower() and "dysample" in model_name.lower()) or "rplksrd" in model_name.lower():
        model_type = "RealPLKSR_dysample"
    elif "realplksr" in model_name.lower() or "rplksr" in model_name.lower():
        model_type = "RealPLKSR"
    elif "drct-l" in model_name.lower():
        model_type = "DRCT-L"
    elif "atd" in model_name.lower():
        model_type = "ATD"
    return f"{model_type}, {model_name}"

typed_upscale_models = {get_model_type(key): value[0] for key, value in upscale_models.items()}


class Upscale:
    def inference(self, img, face_restoration, upscale_model, scale: float, face_detection, outputWithModelName: bool):
        print(img)
        print(face_restoration, upscale_model, scale)
        try:
            self.scale = scale
            self.img_name = os.path.basename(str(img))
            self.basename, self.extension = os.path.splitext(self.img_name)
            
            img = cv2.imdecode(np.fromfile(img, np.uint8), cv2.IMREAD_UNCHANGED) # cv2.imread(img, cv2.IMREAD_UNCHANGED)
            
            self.img_mode = "RGBA" if len(img.shape) == 3 and img.shape[2] == 4 else None
            if len(img.shape) == 2:  # for gray inputs
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            h, w = img.shape[0:2]

            if face_restoration:
                download_from_url(face_models[face_restoration][0], face_restoration, os.path.join("weights", "face"))
                
            modelInUse = ""
            upscale_type = None
            if upscale_model:
                upscale_type, upscale_model = upscale_model.split(", ", 1)
                download_from_url(upscale_models[upscale_model][0], upscale_model, os.path.join("weights", "upscale"))
                modelInUse = f"_{os.path.splitext(upscale_model)[0]}"
            
            netscale = 4
            loadnet = None
            model = None
            is_auto_split_upscale = True
            half = True if torch.cuda.is_available() else False
            if upscale_type:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from basicsr.archs.realplksr_arch import realplksr
                # background enhancer with upscale model
                if upscale_type == "RRDB":
                    netscale = 2 if "x2" in upscale_model else 4
                    num_block = 6 if "6B" in upscale_model else 23
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=num_block, num_grow_ch=32, scale=netscale)
                elif upscale_type == "SRVGG":
                    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
                    netscale = 4
                    num_conv = 16 if "animevideov3" in upscale_model else 32
                    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=num_conv, upscale=netscale, act_type='prelu')
                elif upscale_type == "ESRGAN":
                    netscale = 4
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale)
                    loadnet = {}
                    loadnet_origin = torch.load(os.path.join("weights", "upscale", upscale_model), map_location=torch.device('cpu'), weights_only=True)
                    for key, value in loadnet_origin.items():
                        new_key = key.replace("model.0", "conv_first").replace("model.1.sub.23.", "conv_body.").replace("model.1.sub", "body") \
                            .replace(".0.weight", ".weight").replace(".0.bias", ".bias").replace(".RDB1.", ".rdb1.").replace(".RDB2.", ".rdb2.").replace(".RDB3.", ".rdb3.") \
                            .replace("model.3.", "conv_up1.").replace("model.6.", "conv_up2.").replace("model.8.", "conv_hr.").replace("model.10.", "conv_last.")
                        loadnet[new_key] = value
                elif upscale_type == "DAT":
                    from basicsr.archs.dat_arch import DAT
                    half = False
                    netscale = 4
                    expansion_factor = 2. if "dat2" in upscale_model.lower() else 4.
                    model = DAT(img_size=64, in_chans=3, embed_dim=180, split_size=[8,32], depth=[6,6,6,6,6,6], num_heads=[6,6,6,6,6,6], expansion_factor=expansion_factor, upscale=netscale)
                    # # Speculate on the parameters.
                    # loadnet_origin = torch.load(os.path.join("weights", "upscale", upscale_model), map_location=torch.device('cpu'), weights_only=True)
                    # inferred_params = self.infer_parameters_from_state_dict_for_dat(loadnet_origin, netscale)
                    # for param, value in inferred_params.items():
                    #     print(f"{param}: {value}")
                elif upscale_type == "HAT":
                    half = False
                    netscale = 4
                    import torch.nn.functional as F
                    from basicsr.archs.hat_arch import HAT
                    class HATWithAutoPadding(HAT):
                        def pad_to_multiple(self, img, multiple):
                            """
                            Fill the image to multiples of both width and height as integers.
                            """
                            _, _, h, w = img.shape
                            pad_h = (multiple - h % multiple) % multiple
                            pad_w = (multiple - w % multiple) % multiple

                            # Padding on the top, bottom, left, and right.
                            pad_top = pad_h // 2
                            pad_bottom = pad_h - pad_top
                            pad_left = pad_w // 2
                            pad_right = pad_w - pad_left

                            img_padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
                            return img_padded, (pad_top, pad_bottom, pad_left, pad_right)

                        def remove_padding(self, img, pad_info):
                            """
                            Remove padding and restore to the original size, considering upscaling.
                            """
                            pad_top, pad_bottom, pad_left, pad_right = pad_info

                            # Adjust padding based on upscaling factor
                            pad_top = int(pad_top * self.upscale)
                            pad_bottom = int(pad_bottom * self.upscale)
                            pad_left = int(pad_left * self.upscale)
                            pad_right = int(pad_right * self.upscale)

                            return img[:, :, pad_top:-pad_bottom if pad_bottom > 0 else None, pad_left:-pad_right if pad_right > 0 else None]

                        def forward(self, x):
                            # Step 1: Auto padding
                            x_padded, pad_info = self.pad_to_multiple(x, self.window_size)

                            # Step 2: Normal model processing
                            x_processed = super().forward(x_padded)

                            # Step 3: Remove padding
                            x_cropped = self.remove_padding(x_processed, pad_info)
                            return x_cropped

                    # The parameters are derived from the XPixelGroup project files: HAT-L_SRx4_ImageNet-pretrain.yml and HAT-S_SRx4.yml.
                    # https://github.com/XPixelGroup/HAT/tree/main/options/test
                    if "hat-l" in upscale_model.lower():
                        window_size = 16
                        compress_ratio = 3
                        squeeze_factor = 30
                        depths = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
                        embed_dim = 180
                        num_heads = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
                        mlp_ratio = 2
                        upsampler = "pixelshuffle"
                    elif "hat-s" in upscale_model.lower():
                        window_size = 16
                        compress_ratio = 24
                        squeeze_factor = 24
                        depths = [6, 6, 6, 6, 6, 6]
                        embed_dim = 144
                        num_heads = [6, 6, 6, 6, 6, 6]
                        mlp_ratio = 2
                        upsampler = "pixelshuffle"
                    model = HATWithAutoPadding(img_size=64, patch_size=1, in_chans=3, embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size, compress_ratio=compress_ratio,
                                squeeze_factor=squeeze_factor, conv_scale=0.01, overlap_ratio=0.5, mlp_ratio=mlp_ratio, upsampler=upsampler, upscale=netscale,)
                elif upscale_type == "RealPLKSR_dysample":
                    netscale = 4
                    model = realplksr(dim=64, n_blocks=28, kernel_size=17, split_ratio=0.25, upscaling_factor=netscale, dysample=True)
                elif upscale_type == "RealPLKSR":
                    half = False if "RealPLSKR" in upscale_model else half
                    netscale = 2 if upscale_model.startswith("2x") else 4
                    model = realplksr(dim=64, n_blocks=28, kernel_size=17, split_ratio=0.25, upscaling_factor=netscale)

            
            self.upsampler = None
            if loadnet:
                self.upsampler = RealESRGANer(scale=netscale, loadnet=loadnet, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)
            elif model:
                self.upsampler = RealESRGANer(scale=netscale, model_path=os.path.join("weights", "upscale", upscale_model), model=model, tile=0, tile_pad=10, pre_pad=0, half=half)
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

                    def enhance(self, img, outscale=None):
                        # img: numpy
                        h_input, w_input = img.shape[0:2]
                        pil_img = self.cv2pil(img)
                        pil_img = self.__call__(pil_img)
                        cv_image = self.pil2cv(pil_img)
                        if outscale is not None and outscale != float(netscale):
                            cv_image = cv2.resize(
                                cv_image, (
                                    int(w_input * outscale),
                                    int(h_input * outscale),
                                ), interpolation=cv2.INTER_LANCZOS4)
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

                self.face_enhancer = GFPGANer(model_path=model_path, upscale=self.scale, arch=arch, channel_multiplier=channel_multiplier, bg_upsampler=self.upsampler, model_rootpath=model_rootpath, det_model=face_detection, resolution=resolution)

            files = []
            if not outputWithModelName:
                modelInUse = ""

            try:
                bg_upsample_img = None
                if self.upsampler and self.upsampler.enhance:
                    from utils.dataops import auto_split_upscale
                    bg_upsample_img, _ = auto_split_upscale(img, self.upsampler.enhance, self.scale) if is_auto_split_upscale else self.upsampler.enhance(img, outscale=self.scale)
                    
                if self.face_enhancer:
                    cropped_faces, restored_aligned, bg_upsample_img = self.face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True, bg_upsample_img=bg_upsample_img)
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


    def infer_parameters_from_state_dict_for_dat(self, state_dict, upscale=4):
        if "params" in state_dict:
            state_dict = state_dict["params"]
        elif "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]

        inferred_params = {}

        # Speculate on the depth.
        depth = {}
        for key in state_dict.keys():
            if "blocks" in key:
                layer = int(key.split(".")[1])
                block = int(key.split(".")[3])
                depth[layer] = max(depth.get(layer, 0), block + 1)
        inferred_params["depth"] = [depth[layer] for layer in sorted(depth.keys())]

        # Speculate on the number of num_heads per layer.
        # ex.
        # layers.0.blocks.1.attn.temperature: torch.Size([6, 1, 1])
        # layers.5.blocks.5.attn.temperature: torch.Size([6, 1, 1])
        # The shape of temperature is [num_heads, 1, 1].
        num_heads = []
        for layer in range(len(inferred_params["depth"])):
            for block in range(inferred_params["depth"][layer]):
                key = f"layers.{layer}.blocks.{block}.attn.temperature"
                if key in state_dict:
                    num_heads_layer = state_dict[key].shape[0]
                    num_heads.append(num_heads_layer)
                    break

        inferred_params["num_heads"] = num_heads

        # Speculate on embed_dim.
        # ex. layers.0.blocks.0.attn.qkv.weight: torch.Size([540, 180])
        for key in state_dict.keys():
            if "attn.qkv.weight" in key:
                qkv_weight = state_dict[key]
                embed_dim = qkv_weight.shape[1]  # Note: The in_features of qkv corresponds to embed_dim.
                inferred_params["embed_dim"] = embed_dim
                break

        # Speculate on split_size.
        # ex.
        # layers.0.blocks.0.attn.attns.0.rpe_biases: torch.Size([945, 2])
        # layers.0.blocks.0.attn.attns.0.relative_position_index: torch.Size([256, 256])
        # layers.0.blocks.2.attn.attn_mask_0: torch.Size([16, 256, 256])
        # layers.0.blocks.2.attn.attn_mask_1: torch.Size([16, 256, 256])
        for key in state_dict.keys():
            if "relative_position_index" in key:
                relative_position_size = state_dict[key].shape[0]
                # Determine split_size[0] and split_size[1] based on the provided data.
                split_size_0, split_size_1 = 8, relative_position_size // 8  # 256 = 8 * 32
                inferred_params["split_size"] = [split_size_0, split_size_1]
                break

        # Speculate on the expansion_factor.
        # ex.
        # layers.0.blocks.0.ffn.fc1.weight: torch.Size([360, 180])
        # layers.5.blocks.5.ffn.fc1.weight: torch.Size([360, 180])
        if "embed_dim" in inferred_params:
            for key in state_dict.keys():
                if "ffn.fc1.weight" in key:
                    fc1_weight = state_dict[key]
                    expansion_factor = fc1_weight.shape[0] // inferred_params["embed_dim"]
                    inferred_params["expansion_factor"] = expansion_factor
                    break

        inferred_params["img_size"] = 64
        inferred_params["in_chans"] = 3  # Assume an RGB image.

        for key in state_dict.keys():
            print(f"{key}: {state_dict[key].shape}")

        return inferred_params


    def imwriteUTF8(self, save_path, image): # `cv2.imwrite` does not support writing files to UTF-8 file paths.
        img_name = os.path.basename(save_path)
        _, extension = os.path.splitext(img_name)
        is_success, im_buf_arr = cv2.imencode(extension, image)
        if (is_success): im_buf_arr.tofile(save_path)


def main():
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.975, device='cuda:0')
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
    article = r"""
    [![download](https://img.shields.io/github/downloads/TencentARC/GFPGAN/total.svg)](https://github.com/TencentARC/GFPGAN/releases)
    [![GitHub Stars](https://img.shields.io/github/stars/TencentARC/GFPGAN?style=social)](https://github.com/TencentARC/GFPGAN)
    [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2101.04061)
    """

    upscale = Upscale()

    with gr.Blocks(title = title) as demo:
        gr.Markdown(value=f"<h1 style=\"text-align:center;\">{title}</h1><br>{description}")
        with gr.Row():
            with gr.Column(variant="panel"):
                input_image = gr.Image(type="filepath", label="Input", format="png")
                face_model = gr.Dropdown(list(face_models.keys())+[None], type="value", value='GFPGANv1.4.pth', label='Face Restoration version', info="Face Restoration and RealESR can be freely combined in different ways, or one can be set to \"None\" to use only the other model. Face Restoration is primarily used for face restoration in real-life images, while RealESR serves as a background restoration model.")
                upscale_model = gr.Dropdown(list(typed_upscale_models.keys())+[None], type="value", value='SRVGG, realesr-general-x4v3.pth', label='UpScale version')
                upscale_scale = gr.Number(label="Rescaling factor", value=4)
                face_detection = gr.Dropdown(["retinaface_resnet50", "YOLOv5l", "YOLOv5n"], type="value", value="retinaface_resnet50", label="Face Detection type")
                with_model_name = gr.Checkbox(label="Output image files name with model name", value=True)
                with gr.Row():
                    submit = gr.Button(value="Submit", variant="primary", size="lg")
                    clear = gr.ClearButton(
                        components=[
                            input_image,
                            face_model,
                            upscale_model,
                            upscale_scale,
                            face_detection,
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
        with gr.Row(variant="panel"):
            # Convert to Markdown table
            header = "| Upscale Model Name | Info | Download URL |\n|------------|------|--------------|"
            rows = [
                f"| [{key}]({value[1]}) | " + value[2].replace("\n", "<br>") + f" | [download]({value[0]}) |"
                for key, value in upscale_models.items()
            ]
            markdown_table = header + "\n" + "\n".join(rows)
            gr.Markdown(value=markdown_table)

        submit.click(
            upscale.inference, 
            inputs=[
                input_image,
                face_model,
                upscale_model,
                upscale_scale,
                face_detection,
                with_model_name,
            ],
            outputs=[gallerys, outputs],
        )
    
    demo.queue(default_concurrency_limit=1)
    demo.launch(inbrowser=True)


if __name__ == "__main__":
    main()
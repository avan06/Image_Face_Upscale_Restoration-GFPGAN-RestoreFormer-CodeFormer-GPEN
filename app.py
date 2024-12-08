import os
import gc
import cv2
import requests
import numpy as np
import gradio as gr
import torch
import traceback
from facexlib.utils.misc import download_from_url
from realesrgan.utils import RealESRGANer


# Define URLs and their corresponding local storage paths
face_model = {
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "RestoreFormer++.ckpt": "https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer++.ckpt",
    # "CodeFormer.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/CodeFormer.pth",
    # legacy model
    "GFPGANv1.3.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
    "GFPGANv1.2.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth",
    "RestoreFormer.ckpt": "https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer.ckpt",
}
realesr_model = {
    # SRVGGNet
    "realesr-general-x4v3.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",                # x4 SRVGGNet (S size)
    "realesr-animevideov3.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",                # x4 SRVGGNet (XS size)
    # RRDBNet
    "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",    # x4 RRDBNet with 6 blocks
    "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRNet_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    # ESRGAN(oldRRDB)
    "4x-AnimeSharp.pth": "https://huggingface.co/utnah/esrgan/resolve/main/4x-AnimeSharp.pth?download=true",                                 # https://openmodeldb.info/models/4x-AnimeSharp
    "4x_IllustrationJaNai_V1_ESRGAN_135k.pth": "https://drive.google.com/uc?export=download&confirm=1&id=1qpioSqBkB_IkSBhEAewSSNFt6qgkBimP", # https://openmodeldb.info/models/4x-IllustrationJaNai-V1-DAT2
    # DATNet
    "4xNomos8kDAT.pth": "https://github.com/Phhofm/models/releases/download/4xNomos8kDAT/4xNomos8kDAT.pth",                                  # https://openmodeldb.info/models/4x-Nomos8kDAT
    "4x-DWTP-DS-dat2-v3.pth": "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/4x-DWTP-DS-dat2-v3.pth", # https://openmodeldb.info/models/4x-DWTP-DS-dat2-v3
    "4x_IllustrationJaNai_V1_DAT2_190k.pth": "https://drive.google.com/uc?export=download&confirm=1&id=1qpioSqBkB_IkSBhEAewSSNFt6qgkBimP",   # https://openmodeldb.info/models/4x-IllustrationJaNai-V1-DAT2
    # HAT
    "4xNomos8kSCHAT-L.pth": "https://github.com/Phhofm/models/releases/download/4xNomos8kSCHAT/4xNomos8kSCHAT-L.pth",                        # https://openmodeldb.info/models/4x-Nomos8kSCHAT-L
    "4xNomos8kSCHAT-S.pth": "https://github.com/Phhofm/models/releases/download/4xNomos8kSCHAT/4xNomos8kSCHAT-S.pth",                        # https://openmodeldb.info/models/4x-Nomos8kSCHAT-S
    "4xNomos8kHAT-L_otf.pth": "https://github.com/Phhofm/models/releases/download/4xNomos8kHAT-L_otf/4xNomos8kHAT-L_otf.pth",                # https://openmodeldb.info/models/4x-Nomos8kHAT-L-otf
    # RealPLKSR_dysample
    "4xHFA2k_ludvae_realplksr_dysample.pth": "https://github.com/Phhofm/models/releases/download/4xHFA2k_ludvae_realplksr_dysample/4xHFA2k_ludvae_realplksr_dysample.pth", # https://openmodeldb.info/models/4x-HFA2k-ludvae-realplksr-dysample
    "4xArtFaces_realplksr_dysample.pth": "https://github.com/Phhofm/models/releases/download/4xArtFaces_realplksr_dysample/4xArtFaces_realplksr_dysample.pth", # https://openmodeldb.info/models/4x-ArtFaces-realplksr-dysample
    "4x-PBRify_RPLKSRd_V3.pth": "https://github.com/Kim2091/Kim2091-Models/releases/download/4x-PBRify_RPLKSRd_V3/4x-PBRify_RPLKSRd_V3.pth", # https://openmodeldb.info/models/4x-PBRify-RPLKSRd-V3
    "4xNomos2_realplksr_dysample.pth": "https://github.com/Phhofm/models/releases/download/4xNomos2_realplksr_dysample/4xNomos2_realplksr_dysample.pth", # https://openmodeldb.info/models/4x-Nomos2-realplksr-dysample
    # RealPLKSR
    "2x-AnimeSharpV2_RPLKSR_Sharp.pth": "https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV2_Set/2x-AnimeSharpV2_RPLKSR_Sharp.pth", # https://openmodeldb.info/models/2x-AnimeSharpV2-RPLKSR-Sharp
    "2x-AnimeSharpV2_RPLKSR_Soft.pth": "https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV2_Set/2x-AnimeSharpV2_RPLKSR_Soft.pth", # https://openmodeldb.info/models/2x-AnimeSharpV2-RPLKSR-Soft
    "4xPurePhoto-RealPLSKR.pth": "https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/4xPurePhoto-RealPLSKR.pth", # https://openmodeldb.info/models/4x-PurePhoto-RealPLSKR
    "2x_Text2HD_v.1-RealPLKSR.pth": "https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/2x_Text2HD_v.1-RealPLKSR.pth",  # https://openmodeldb.info/models/2x-Text2HD-v-1
    "2xVHS2HD-RealPLKSR.pth": "https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/2xVHS2HD-RealPLKSR.pth", # https://openmodeldb.info/models/2x-VHS2HD
    "4xNomosWebPhoto_RealPLKSR.pth": "https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_RealPLKSR/4xNomosWebPhoto_RealPLKSR.pth", # https://openmodeldb.info/models/4x-NomosWebPhoto-RealPLKSR
}

files_to_download = {
    "a1.jpg":
    "https://thumbs.dreamstime.com/b/tower-bridge-traditional-red-bus-black-white-colors-view-to-tower-bridge-london-black-white-colors-108478942.jpg",
    "a2.jpg":
    "https://media.istockphoto.com/id/523514029/photo/london-skyline-b-w.jpg?s=612x612&w=0&k=20&c=kJS1BAtfqYeUDaORupj0sBPc1hpzJhBUUqEFfRnHzZ0=",
    "a3.jpg":
    "https://i.guim.co.uk/img/media/06f614065ed82ca0e917b149a32493c791619854/0_0_3648_2789/master/3648.jpg?width=700&quality=85&auto=format&fit=max&s=05764b507c18a38590090d987c8b6202",
    "a4.jpg":
    "https://i.pinimg.com/736x/46/96/9e/46969eb94aec2437323464804d27706d--victorian-london-victorian-era.jpg",
}

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
    return f"{model_type}, {model_name}"

typed_realesr_model = {get_model_type(key): value for key, value in realesr_model.items()}

def download_from_urls(urls, save_dir=None):
    for file_name, url in urls.items():
        download_from_url(url, file_name, save_dir)


class Upscale:
    def inference(self, img, face_restoration, realesr, scale: float):
        print(img)
        print(face_restoration, realesr, scale)
        try:
            self.scale = scale
            self.img_name = os.path.basename(str(img))
            self.basename, self.extension = os.path.splitext(self.img_name)
            
            img = cv2.imdecode(np.fromfile(img, np.uint8), cv2.IMREAD_UNCHANGED) # cv2.imread(img, cv2.IMREAD_UNCHANGED)
            
            self.img_mode = "RGBA" if len(img.shape) == 3 and img.shape[2] == 4 else None
            if len(img.shape) == 2:  # for gray inputs
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            h, w = img.shape[0:2]
            if h < 300:
                img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

            if face_restoration:
                download_from_url(face_model[face_restoration], face_restoration, os.path.join("weights", "face"))
            if realesr:
                realesr_type, realesr = realesr.split(", ", 1)
                download_from_url(realesr_model[realesr], realesr, os.path.join("weights", "realesr"))
            
            netscale = 4
            loadnet = None
            model = None
            is_auto_split_upscale = True
            half = True if torch.cuda.is_available() else False
            if realesr_type:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from basicsr.archs.realplksr_arch import realplksr
                # background enhancer with RealESRGAN
                if realesr_type == "RRDB":
                    netscale = 2 if "x2" in realesr else 4
                    num_block = 6 if "6B" in realesr else 23
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=num_block, num_grow_ch=32, scale=netscale)
                elif realesr_type == "SRVGG":
                    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
                    netscale = 4
                    num_conv = 16 if "animevideov3" in realesr else 32
                    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=num_conv, upscale=netscale, act_type='prelu')
                elif realesr_type == "ESRGAN":
                    netscale = 4
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale)
                    loadnet = {}
                    loadnet_origin = torch.load(os.path.join("weights", "realesr", realesr), map_location=torch.device('cpu'), weights_only=True)
                    for key, value in loadnet_origin.items():
                        new_key = key.replace("model.0", "conv_first").replace("model.1.sub.23.", "conv_body.").replace("model.1.sub", "body") \
                            .replace(".0.weight", ".weight").replace(".0.bias", ".bias").replace(".RDB1.", ".rdb1.").replace(".RDB2.", ".rdb2.").replace(".RDB3.", ".rdb3.") \
                            .replace("model.3.", "conv_up1.").replace("model.6.", "conv_up2.").replace("model.8.", "conv_hr.").replace("model.10.", "conv_last.")
                        loadnet[new_key] = value
                elif realesr_type == "DAT":
                    from basicsr.archs.dat_arch import DAT
                    half = False
                    netscale = 4
                    expansion_factor = 2. if "dat2" in realesr.lower() else 4.
                    model = DAT(img_size=64, in_chans=3, embed_dim=180, split_size=[8,32], depth=[6,6,6,6,6,6], num_heads=[6,6,6,6,6,6], expansion_factor=expansion_factor, upscale=netscale)
                    # # Speculate on the parameters.
                    # loadnet_origin = torch.load(os.path.join("weights", "realesr", realesr), map_location=torch.device('cpu'), weights_only=True)
                    # inferred_params = self.infer_parameters_from_state_dict_for_dat(loadnet_origin, netscale)
                    # for param, value in inferred_params.items():
                    #     print(f"{param}: {value}")
                elif realesr_type == "HAT":
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
                    if "hat-l" in realesr.lower():
                        window_size = 16
                        compress_ratio = 3
                        squeeze_factor = 30
                        depths = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
                        embed_dim = 180
                        num_heads = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
                        mlp_ratio = 2
                        upsampler = "pixelshuffle"
                    elif "hat-s" in realesr.lower():
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
                elif realesr_type == "RealPLKSR_dysample":
                    netscale = 4
                    model = realplksr(upscaling_factor=netscale, dysample=True)
                elif realesr_type == "RealPLKSR":
                    half = False if "RealPLSKR" in realesr else half
                    netscale = 2 if realesr.startswith("2x") else 4
                    model = realplksr(dim=64, n_blocks=28, kernel_size=17, split_ratio=0.25, upscaling_factor=netscale)

            
            self.upsampler = None
            if loadnet:
                self.upsampler = RealESRGANer(scale=netscale, loadnet=loadnet, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)
            elif model:
                self.upsampler = RealESRGANer(scale=netscale, model_path=os.path.join("weights", "realesr", realesr), model=model, tile=0, tile_pad=10, pre_pad=0, half=half)
            elif realesr:
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
                upscaler = UpscaleWithModel.from_pretrained(os.path.join("weights", "realesr", realesr)).to(device)
                upscaler.__class__ = UpscaleWithModel_Gfpgan
                self.upsampler = upscaler
            self.face_enhancer = None

            if face_restoration:
                from gfpgan.utils import GFPGANer
                if face_restoration and face_restoration.startswith("GFPGANv1."):
                    self.face_enhancer = GFPGANer(model_path=os.path.join("weights", "face", face_restoration), upscale=self.scale, arch="clean", channel_multiplier=2, bg_upsampler=self.upsampler)
                elif face_restoration and face_restoration.startswith("RestoreFormer"):
                    arch = "RestoreFormer++" if face_restoration.startswith("RestoreFormer++") else "RestoreFormer"
                    self.face_enhancer = GFPGANer(model_path=os.path.join("weights", "face", face_restoration), upscale=self.scale, arch=arch, channel_multiplier=2, bg_upsampler=self.upsampler)
                elif face_restoration == 'CodeFormer.pth':
                     self.face_enhancer = GFPGANer(
                     model_path='weights/CodeFormer.pth', upscale=self.scale, arch='CodeFormer', channel_multiplier=2, bg_upsampler=self.upsampler)


            files = []
            outputs = []
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
                            save_crop_path = f"output/{self.basename}{idx:02d}_cropped_faces.png"
                            self.imwriteUTF8(save_crop_path, cropped_face)
                            # save restored face
                            save_restore_path = f"output/{self.basename}{idx:02d}_restored_faces.png"
                            self.imwriteUTF8(save_restore_path, restored_face)
                            # save comparison image
                            save_cmp_path = f"output/{self.basename}{idx:02d}_cmp.png"
                            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                            self.imwriteUTF8(save_cmp_path, cmp_img)
        
                            files.append(save_crop_path)
                            files.append(save_restore_path)
                            files.append(save_cmp_path)
                            outputs.append(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
                            outputs.append(cv2.cvtColor(restored_face, cv2.COLOR_BGR2RGB))
                            outputs.append(cv2.cvtColor(cmp_img, cv2.COLOR_BGR2RGB))

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
            save_path = f"output/{self.basename}{self.extension}"
            self.imwriteUTF8(save_path, restored_img)

            restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
            files.append(save_path)
            outputs.append(restored_img)
            return outputs, files
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

    # Iterate through each file
    download_from_urls(files_to_download, ".")

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

    upscale = Upscale()

    demo = gr.Interface(
        upscale.inference, [
            gr.Image(type="filepath", label="Input", format="png"),
            gr.Dropdown(list(face_model.keys())+[None], type="value", value='GFPGANv1.4.pth', label='Face Restoration version', info="Face Restoration and RealESR can be freely combined in different ways, or one can be set to \"None\" to use only the other model. Face Restoration is primarily used for face restoration in real-life images, while RealESR serves as a background restoration model."),
            gr.Dropdown(list(typed_realesr_model.keys())+[None], type="value", value='SRVGG, realesr-general-x4v3.pth', label='RealESR version'),
            gr.Number(label="Rescaling factor", value=4),
        ], [
            gr.Gallery(type="numpy", label="Output (The whole image)", format="png"),
            gr.File(label="Download the output image")
        ],
        title=title,
        description=description,
        article=article,
        examples=[["a1.jpg", "GFPGANv1.4.pth", "SRVGG, realesr-general-x4v3.pth", 2], 
                  ["a2.jpg", "GFPGANv1.4.pth", "SRVGG, realesr-general-x4v3.pth", 2], 
                  ["a3.jpg", "GFPGANv1.4.pth", "SRVGG, realesr-general-x4v3.pth", 2],
                  ["a4.jpg", "GFPGANv1.4.pth", "SRVGG, realesr-general-x4v3.pth", 2]])
    
    demo.queue(default_concurrency_limit=4)
    demo.launch(inbrowser=True)


if __name__ == "__main__":
    main()
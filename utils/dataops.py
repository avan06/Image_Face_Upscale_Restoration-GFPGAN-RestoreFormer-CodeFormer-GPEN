#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# The file source is from the [ESRGAN](https://github.com/xinntao/ESRGAN) project 
# forked by authors [joeyballentine](https://github.com/joeyballentine/ESRGAN) and [BlueAmulet](https://github.com/BlueAmulet/ESRGAN).

import gc

import numpy as np
import torch


def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    # flip image channels
    # https://github.com/pytorch/pytorch/issues/229
    out: torch.Tensor = image.flip(-3)
    # out: torch.Tensor = image[[2, 1, 0], :, :] #RGB to BGR #may be faster
    return out


def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    # same operation as bgr_to_rgb(), flip image channels
    return bgr_to_rgb(image)


def bgra_to_rgba(image: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = image[[2, 1, 0, 3], :, :]
    return out


def rgba_to_bgra(image: torch.Tensor) -> torch.Tensor:
    # same operation as bgra_to_rgba(), flip image channels
    return bgra_to_rgba(image)


def auto_split_upscale(
    lr_img: np.ndarray,
    upscale_function,
    scale: int = 4,
    overlap: int = 32,
    max_depth: int = None,
    current_depth: int = 1,
    current_tile: int = 1,  # Tracks the current tile being processed
    total_tiles: int = 1,  # Total number of tiles at this depth level
):
    # Attempt to upscale if unknown depth or if reached known max depth
    if max_depth is None or max_depth == current_depth:
        try:
            print(f"auto_split_upscale depth: {current_depth}", end=" ", flush=True)
            result, _ = upscale_function(lr_img, scale)
            print(f"progress: {current_tile}/{total_tiles}")
            return result, current_depth
        except RuntimeError as e:
            # Check to see if its actually the CUDA out of memory error
            if "CUDA" in str(e):
                print("RuntimeError: CUDA out of memory...")
            # Re-raise the exception if not an OOM error
            else:
                raise RuntimeError(e)
            # Collect garbage (clear VRAM)
            torch.cuda.empty_cache()
            gc.collect()

    input_h, input_w, input_c = lr_img.shape
    
    # Split the image into 4 quadrants with some overlap
    top_left      = lr_img[: input_h // 2 + overlap, : input_w // 2 + overlap, :]
    top_right     = lr_img[: input_h // 2 + overlap, input_w // 2 - overlap :, :]
    bottom_left   = lr_img[input_h // 2 - overlap :, : input_w // 2 + overlap, :]
    bottom_right  = lr_img[input_h // 2 - overlap :, input_w // 2 - overlap :, :]
    current_depth = current_depth + 1
    current_tile  = (current_tile - 1) * 4
    total_tiles   = total_tiles * 4

    # Recursively upscale each quadrant and track the current tile number
    # After we go through the top left quadrant, we know the maximum depth and no longer need to test for out-of-memory
    top_left_rlt, depth = auto_split_upscale(
        top_left, upscale_function, scale=scale, overlap=overlap, max_depth=max_depth,
        current_depth=current_depth, current_tile=current_tile + 1, total_tiles=total_tiles,
    )
    top_right_rlt, _ = auto_split_upscale(
        top_right, upscale_function, scale=scale, overlap=overlap, max_depth=depth,
        current_depth=current_depth, current_tile=current_tile + 2, total_tiles=total_tiles,
    )
    bottom_left_rlt, _ = auto_split_upscale(
        bottom_left, upscale_function, scale=scale, overlap=overlap, max_depth=depth,
        current_depth=current_depth, current_tile=current_tile + 3, total_tiles=total_tiles,
    )
    bottom_right_rlt, _ = auto_split_upscale(
        bottom_right, upscale_function, scale=scale, overlap=overlap, max_depth=depth,
        current_depth=current_depth, current_tile=current_tile + 4, total_tiles=total_tiles,
    )
    
    # Define the output image size
    out_h = input_h * scale
    out_w = input_w * scale
    
    # Create an empty output image
    output_img = np.zeros((out_h, out_w, input_c), np.uint8)

    # Fill the output image with the upscaled quadrants, removing overlap regions
    output_img[: out_h // 2, : out_w // 2, :]   = top_left_rlt[: out_h // 2, : out_w // 2, :]
    output_img[: out_h // 2, -out_w // 2 :, :]  = top_right_rlt[: out_h // 2, -out_w // 2 :, :]
    output_img[-out_h // 2 :, : out_w // 2, :]  = bottom_left_rlt[-out_h // 2 :, : out_w // 2, :]
    output_img[-out_h // 2 :, -out_w // 2 :, :] = bottom_right_rlt[-out_h // 2 :, -out_w // 2 :, :]

    return output_img, depth

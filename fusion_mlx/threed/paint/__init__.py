# SPDX-License-Identifier: Apache-2.0
# Hunyuan3D-2.1 MLX-native texture paint port (issue #989 Session 4).
# Dual-branch (albedo + mr) multiview diffusion UNet + VAE + DINOv2-Giant
# conditioner + image_proj + DDIM v_prediction scheduler + mesh multiview
# rasterizer + xatlas UV unwrap + texture bake-back -> textured GLB.
# Weights: paint/{unet,vae,dino}.safetensors (mixed fp16 + 8bit quant group 16).
# No zig reference (paint is "a later phase" per src/hunyuan3d.zig comment) —
# ported from config + checkpoint key layout + diffusers UNet2DConditionModel
# conventions, validated by real-weight load parity.

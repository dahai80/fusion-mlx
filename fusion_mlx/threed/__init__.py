# SPDX-License-Identifier: Apache-2.0
# Hunyuan3D-2.1 MLX-native 3D generation port (issue #989).
# Multi-session plan:
#   Session 1 = DINOv2 conditioner + config parse + smoke load (landed).
#   Session 2 = ShapeVAE decoder + geo SDF decoder + marching cubes mesh (landed).
#   Session 3 = flow-match MoE DiT denoiser (landed).
# Stages remaining: texture paint UNet (S4), /v1/3d/generate API (S5).
# Weights: ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit (MLX-native 8bit, tencent license).

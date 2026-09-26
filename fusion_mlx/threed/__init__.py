# SPDX-License-Identifier: Apache-2.0
# Hunyuan3D-2.1 MLX-native 3D generation port (issue #989).
# Multi-session plan: Session 1 = DINOv2 conditioner + config parse + smoke load.
# Stages: shape (DiT+VAE), texture (paint UNet+SD2 VAE), auto-rig (UniRig).
# Weights: ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit (MLX-native 8bit, tencent license).

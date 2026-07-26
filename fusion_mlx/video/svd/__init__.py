# SPDX-License-Identifier: Apache-2.0
# Pure-MLX port of Stable Video Diffusion (SVD) img2vid-xt.
# I2V only: encodes input image via CLIP vision + VAE, then denoises
# with a temporal UNet using Euler v-prediction scheduling.

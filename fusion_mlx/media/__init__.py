# SPDX-License-Identifier: Apache-2.0
# Media subprocess isolation (S3, audit 0910 §6.2).
# Image generation runs in a subprocess so its Metal allocations cannot
# crash the LLM in the main process. Worker loads model, generates, writes
# output, exits — memory fully released on exit.

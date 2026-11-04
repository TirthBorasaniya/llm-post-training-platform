# LLM Post-Training and Inference Platform

A reproducible platform that takes an open language model through alignment, distillation, quantization, and serving, with every stage evaluated. The repository contains the training, evaluation, distillation, quantization, serving, benchmarking, and deployment code. Heavy stages run on a SLURM cluster or on a single GPU machine, selected in `configs/compute.yaml`.

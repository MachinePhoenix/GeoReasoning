# Gemma3 SFT + RAFT

We modify LLaMA-Factory(https://github.com/hiyouga/LLaMA-Factory) for SFT and vllm infer.

"scripts_raft" folder contains some shell scripts related to the RAFT process under different settings, and you can refer to the .py files. These files are mainly in the "src_raft" folder. But some of them (for example, 'caption_generation_llamafactory_ray.py') are called under the path "LLaMA-Factory/scripts", although we also copy them into the "src_raft" folder.

"scripts_eval" folder contains some shell scripts related to evaluation on downstream benchmarks (MathVista and MathVerse), and you can refer to the called .py files (mainly in the "src_eval" folder).

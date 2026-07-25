"""Defense module: two-stage iterative safety fine-tuning.

Pipeline: clean / under-test / malicious sample construction → stage-1 alignment
SFT → automatic risk re-labeling of under-test samples → stage-2 SFT, looped as
"generate → screen → evaluate → repair".
"""

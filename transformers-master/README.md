
We adapt our code from huggingface. For details, please refer to the [original repository](https://github.com/huggingface/transformers).

Our code only works for ALBERT up to now. The codebase is compatible for finetuning pretrained ALBERT model on GLUE and 
SQUAD datasets. We provide an example bash file run_glue_albert_mrpc.sh under examples folder to finetune on MRPC dataset. The flag
att_type controls whether deterministic attention or stochastic attention was used. 


## Citation

```bibtex

@inproceedings{Fan2020bayesian,
  title={Bayesian Attention Modules},
  author={Xinjie Fan and Shujian Zhang and Bo Chen and Mingyuan Zhou},
  booktitle = {NeurIPS 2020: Advances in Neural Information Processing Systems},
  month={Dec.},
  year = {2020}
}
```

---
library_name: transformers
license: mit
base_model: prajjwal1/bert-tiny
tags:
- generated_from_trainer
metrics:
- accuracy
- f1
- precision
- recall
model-index:
- name: tiny-sentiment
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# tiny-sentiment

This model is a fine-tuned version of [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6871
- Accuracy: 0.56
- F1: 0.0
- Precision: 0.0
- Recall: 0.0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- optimizer: Use adamw_torch_fused with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 1

### Training results



### Framework versions

- Transformers 4.56.0
- Pytorch 2.8.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.0

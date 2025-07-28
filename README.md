-whisper-small-monglian-commonvioce
Whisper Small Fine - Tuned with Mongolian Speech Dataset
# Whisper Small Fine-tuned with Mongolian Speech Dataset

This model is a fine-tuned version of openai/whisper-small on the Mongolian Speech Dataset dataset.
It achieves the following results on the test set:

| Metric         | Value     |
|----------------|-----------|
| Loss           | 0.0192 |
| RUNTIME           | 138.3573 |
| SAMPLES_PER_SECOND           | 13.9780 |
| STEPS_PER_SECOND           | 0.8750 |

## Training results

| Training Loss | Epoch   | Step  | Validation Loss | RUNTIME | SAMPLES_PER_SECOND | STEPS_PER_SECOND |
|---------------|---------|-------|----------------- |-------- |-------- |-------- |
| 1.9226 | 1.0759 | 100 | - | - | - | - |
| 0.5600 | 2.1519 | 200 | - | - | - | - |
| 0.3869 | 3.2278 | 300 | - | - | - | - |
| - | 4.3038 | 400 | 0.3193 | 141.4786 | 13.6700 | 0.8550 |
| 0.2569 | 5.3797 | 500 | - | - | - | - |
| 0.2190 | 6.4557 | 600 | - | - | - | - |
| 0.1879 | 7.5316 | 700 | - | - | - | - |
| - | 8.6076 | 800 | 0.1808 | 139.3554 | 13.8780 | 0.8680 |
| 0.1427 | 9.6835 | 900 | - | - | - | - |
| 0.1248 | 10.7595 | 1000 | - | - | - | - |
| 0.1097 | 11.8354 | 1100 | - | - | - | - |
| - | 12.9114 | 1200 | 0.1139 | 140.4444 | 13.7710 | 0.8620 |
| 0.0857 | 13.9873 | 1300 | - | - | - | - |
| 0.0760 | 15.0542 | 1400 | - | - | - | - |
| 0.0684 | 16.1302 | 1500 | - | - | - | - |
| - | 17.2061 | 1600 | 0.0739 | 140.8514 | 13.7310 | 0.8590 |
| 0.0536 | 18.2821 | 1700 | - | - | - | - |
| 0.0481 | 19.3580 | 1800 | - | - | - | - |
| 0.0430 | 20.4340 | 1900 | - | - | - | - |
| - | 21.5099 | 2000 | 0.0457 | 140.6019 | 13.7550 | 0.8610 |
| 0.0353 | 22.5859 | 2100 | - | - | - | - |
| 0.0316 | 23.6618 | 2200 | - | - | - | - |
| 0.0295 | 24.7378 | 2300 | - | - | - | - |
| - | 25.8137 | 2400 | 0.0362 | 140.6972 | 13.7460 | 0.8600 |
| 0.0239 | 26.8897 | 2500 | - | - | - | - |
| 0.0221 | 27.9656 | 2600 | - | - | - | - |
| 0.0200 | 29.0325 | 2700 | - | - | - | - |
| - | 30.1085 | 2800 | 0.0271 | 140.8299 | 13.7330 | 0.8590 |
| 0.0171 | 31.1844 | 2900 | - | - | - | - |
| 0.0165 | 32.2604 | 3000 | - | - | - | - |
| 0.0147 | 33.3363 | 3100 | - | - | - | - |
| - | 34.4123 | 3200 | 0.0192 | 138.3573 | 13.9780 | 0.8750 |
| 0.0133 | 35.4882 | 3300 | - | - | - | - |
| 0.0124 | 36.5642 | 3400 | - | - | - | - |
| 0.0118 | 37.6401 | 3500 | - | - | - | - |

## Training hyperparameters
The following hyperparameters were used during training:

- learning_rate: 1.5e-5
- train_batch_size: 16
- eval_batch_size: 16
- gradient_accumulation_steps: 12
- total_train_batch_size: 192
- optimizer: AdamW with betas=(0.9, 0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 400
- training_steps: 3500
- mixed_precision_training: FP16

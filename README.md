Whisper Small Fine - Tuned with Mongolian Speech Dataset
Overview
This repository presents a fine - tuned version of the openai/whisper-small model specifically tailored to the Mongolian Speech Dataset. The fine - tuning process has significantly enhanced the model's performance in transcribing Mongolian speech, making it a valuable asset for applications in Mongolian language processing.
Performance Metrics
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

- Training Hyperparameters
The following hyperparameters were carefully selected and used during the training process:
Learning Rate: 1.5e - 5
This learning rate was chosen to ensure a stable and efficient learning process. A too - large learning rate may cause the model to diverge, while a too - small one may lead to slow convergence.
Train Batch Size: 16
The batch size affects the efficiency of the training process and the generalization ability of the model. A batch size of 16 was found to be optimal for this task.
Eval Batch Size: 16
Consistent with the train batch size, it ensures fair evaluation during the training process.
Gradient Accumulation Steps: 12
This technique allows for effective use of limited GPU memory by accumulating gradients over multiple steps before performing an optimization step.
Total Train Batch Size: 192
Calculated based on the train batch size and gradient accumulation steps, it reflects the overall batch size for each optimization step.
Optimizer: AdamW with betas=(0.9, 0.999) and epsilon = 1e - 08
AdamW is a widely used optimizer that combines the advantages of Adam and weight decay, which helps prevent overfitting.
LR Scheduler Type: linear
A linear learning rate scheduler gradually decreases the learning rate over training steps, which is beneficial for the model to converge to a better solution.
LR Scheduler Warmup Steps: 400
The warmup steps help the model adapt to the data in the early stages of training and avoid large fluctuations in the learning process.
Training Steps: 3500
This number of training steps was determined based on the model's performance on the validation set, ensuring sufficient training without overfitting.
Mixed Precision Training: FP16
Mixed precision training using FP16 significantly reduces the memory usage and speeds up the training process without sacrificing much accuracy.
Applications
This fine - tuned model can be applied in various Mongolian language - related fields, such as:
Speech Recognition Systems: Providing accurate transcription of Mongolian speech in real - time or offline scenarios.
Language Learning Platforms: Helping learners improve their listening and speaking skills in Mongolian.
Accessibility Tools: Making Mongolian audio content more accessible to people with hearing impairments.
Future Work
Further Fine - Tuning: Explore additional data sources and fine - tuning techniques to further improve the model's performance.
Multilingual Integration: Integrate the model with other language models to support multilingual speech processing.
Deployment Optimization: Optimize the model for deployment on different platforms, such as mobile devices and cloud servers.
This project showcases my skills in model fine - tuning, hyperparameter optimization, and performance evaluation, which are essential for developing high - quality deep learning models.

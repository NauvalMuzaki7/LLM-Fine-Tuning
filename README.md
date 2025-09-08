# Fine-Tuning LLM for Task-Oriented Instruction Generation

This repository contains a complete solution for the **ML Engineer Take-Home Test**, focusing on *fine-tuning* a Large Language Model (LLM) to generate structured, task-oriented instructions from natural language user input.

---

## 1. Selection of Open-Source LLM

**Chosen Model:** `mistralai/Mistral-7B-Instruct-v0.2`

### Justification
- **Instruction-Focused Architecture**  
  The *Instruct* version of Mistral is explicitly optimized for *instruction following* and dialogue, making it highly effective at producing accurate and coherent numbered instruction lists.
  
- **Performance vs. Efficiency Balance**  
  Comparable to larger models like Llama 2 13B, while being more lightweight (7B parameters). With QLoRA, fine-tuning can be performed on a single consumer GPU (RTX 3090/4090).

- **Flexible Licensing**  
  Released under **Apache 2.0** license → allows commercial usage.

- **Mature Ecosystem and Support**  
  Fully integrated with Hugging Face (`transformers`, `peft`, `trl`) and backed by an active community.

---

## 2. Dataset Design and Preparation

### Data Type and Format
The dataset is stored in **JSONL** format, where each line contains a `prompt`–`output` pair.

**Example:**
```json
{"prompt": "I forgot my Gojek account password, how can I reset it?", "output": "1. Open the Gojek app.\n2. On the login screen, tap 'Forgot password?'.\n3. Enter your registered phone number or email.\n4. Tap 'Continue'.\n5. Check your email or SMS for the reset link.\n6. Click the link and set your new password."}
{"prompt": "How to change profile picture on Instagram?", "output": "1. Open the Instagram app and go to your profile page.\n2. Tap 'Edit Profile'.\n3. Tap 'Change Profile Picture'.\n4. Select 'New Profile Photo' to choose from your gallery.\n5. Adjust the photo and tap 'Done'."}
```

---

## Handling Edge Cases & Bias

- **Data Imbalance** → addressed with augmentation and oversampling.  
- **Diversity** → ensure coverage across multiple domains (e-commerce, finance, social media) and language styles.  

---

## 3. Fine-Tuning Strategy

**Approach:** `QLoRA (Quantized Low-Rank Adaptation)`

### Why QLoRA?
- **Memory Efficiency** → model loaded in 4-bit precision.  
- **High Quality without Compromise** → near full fine-tuning performance.  
- **Training Stability** → prevents *catastrophic forgetting*.  

### Key Hyperparameters
- `learning_rate`: 2e-4 (AdamW)  
- `lora_r`: 16–64  
- `lora_alpha`: 2 × r  
- `per_device_train_batch_size`: depends on available VRAM  
- `num_train_epochs`: 1–3  

### Challenge Mitigation
- **Overfitting** → validation set + *early stopping* + `lora_dropout`.  
- **Compute Constraints** → addressed via QLoRA.  
- **Quality Evaluation** → mix of automated metrics and human judgment.  

---

## 4. Evaluation and Benchmarking

### Quantitative Metrics
- **BERTScore** → evaluates semantic similarity.  
- **ROUGE-L** → evaluates structural similarity of instructions.  

### Qualitative Evaluation
Human reviewers assess model outputs based on:  
- **Accuracy** → Are the steps factually correct?  
- **Completeness** → Are critical steps missing?  
- **Clarity** → Is the language clear and easy to follow?  
- **Format Adherence** → Does the model consistently produce cleanly numbered lists?  

### Benchmark Setup
Comparisons are made across three baselines:  
1. **Baseline:** Mistral-7B-Instruct-v0.2 (no fine-tuning).  
2. **Fine-Tuned Model:** QLoRA-adapted version.  
3. **Gold Standard:** human-annotated answers in the test set.  

Evaluation is conducted using a **15% held-out test set** + *blind test* human review to avoid bias.  

---

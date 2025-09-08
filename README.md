# Fine-Tuning LLM for Task-Oriented Instruction Generation

This repository contains a complete solution for the ML Engineer Take-Home Test, focusing on fine-tuning a Large Language Model (LLM) to generate structured and task-oriented instructions from natural language user input.

---

## 1. Selection of Open-Source LLM

**Chosen Model:** `mistralai/Mistral-7B-Instruct-v0.2`

For this task, I chose **Mistral-7B-Instruct-v0.2**. This model strikes a balance between high performance and resource efficiency, making it an ideal candidate for fine-tuning on specific use cases like this.

### Justification

- **Instruction-Optimized Architecture**  
  The *Instruct* version of Mistral has been explicitly optimized for instruction following and dialogue. This gives it strong foundational capabilities for structured formatting, which is critical for generating accurate and coherent numbered instruction lists.

- **Performance vs. Efficiency Balance**  
  Mistral-7B matches or even surpasses larger models (e.g., Llama 2 13B) across benchmarks. Its 7B parameter size allows fine-tuning on a single consumer-grade GPU (such as NVIDIA RTX 3090/4090) using **QLoRA**, reducing costs and development time significantly.

- **Flexible Licensing**  
  Released under the **Apache 2.0 license**, the model permits commercial use. This is crucial for scalability and integration into real-world products without restrictive licensing barriers.

- **Mature Ecosystem and Support**  
  Mistral-7B is fully integrated into the Hugging Face ecosystem, including `transformers`, `peft`, and `trl`. Rich documentation and an active community accelerate implementation and troubleshooting.

---

## 2. Dataset Design and Preparation

The quality of the fine-tuned model heavily depends on the training dataset. I propose a multi-step approach for dataset design and preparation.

### Data Type and Format

The dataset will be structured in **JSONL** format, where each line is a JSON object containing a user prompt and the ideal instruction output.  

**Example:**

```json
{"prompt": "I forgot my Gojek account password, how can I reset it?", "output": "1. Open the Gojek app.\n2. On the login page, tap 'Forgot password?'.\n3. Enter your registered phone number or email.\n4. Tap 'Continue'.\n5. Check your email or SMS for the reset password link.\n6. Click the link and set your new password."}

{"prompt": "how to change profile picture on instagram?", "output": "1. Open the Instagram app and go to your profile.\n2. Tap 'Edit Profile'.\n3. Tap 'Change Profile Picture'.\n4. Choose 'New Profile Photo' from your gallery.\n5. Adjust the photo and tap 'Done'."}
```

## Data Collection and Annotation

A hybrid strategy will be used to ensure a diverse, high-quality dataset:

### Synthetic Generation with Supervision
Use advanced models (e.g., GPT-4o) to generate thousands of initial samples with prompt engineering for task variety.

### Human Curation
Human annotators will review, correct, and validate synthetic data to ensure factual accuracy, clarity, and completeness.

### Real-World Extraction
Collect data from product manuals, FAQs, and support forums to ensure realistic coverage of use cases.

---

## Preprocessing Steps

- **Cleaning and Normalization**: Remove duplicates, fix typos, and standardize formats (e.g., ensure numbered instructions).  
- **PII Anonymization**: Detect and mask personal data (names, emails, phone numbers).  
- **Instruction Formatting Template**: Use Mistral’s instruct format:  


---

## Handling Edge Cases and Bias

- **Data Imbalance**: Apply augmentation (e.g., paraphrasing with other LLMs) and oversampling for rare tasks.  
- **Diversity**: Ensure coverage across domains (e-commerce, finance, social media) and demographics (formal vs. informal language) to improve generalization.

---

## 3. Fine-Tuning Strategy

**Approach:** QLoRA (Quantized Low-Rank Adaptation)  

QLoRA is a variant of LoRA under PEFT (Parameter-Efficient Fine-Tuning).

### Why QLoRA?

- **Maximum Memory Efficiency**  
Loads the base model in 4-bit precision while training low-rank adapters. Enables fine-tuning a 7B model on a single 16GB GPU.  

- **Quality Retention**  
Maintains performance close to full 16-bit fine-tuning.  

- **Training Stability**  
Prevents catastrophic forgetting by freezing most base parameters.  

### Key Hyperparameters

- `learning_rate`: Start with **2e-4** (AdamW optimizer).  
- `lora_r`: Rank (16–64) for LoRA matrices. Higher rank captures complexity but risks overfitting.  
- `lora_alpha`: Typically set to **2 × r**.  
- `per_device_train_batch_size`: Adjusted based on VRAM.  
- `num_train_epochs`: Usually **1–3 epochs** for fine-tuning.  

### Challenges & Mitigations

- **Overfitting**: Use validation set monitoring, early stopping, and LoRA dropout (`lora_dropout`).  
- **Compute Needs**: Addressed via QLoRA’s memory efficiency.  
- **Evaluation Quality**: Human-in-the-loop evaluation complements automatic metrics.  

---

## 4. Evaluation and Benchmarking

Evaluation combines **automatic metrics** with **human judgment** for a holistic performance view.

### Quantitative Metrics

- **BERTScore**: Captures semantic similarity, handling synonyms and paraphrasing better than BLEU/ROUGE.  
- **ROUGE-L**: Measures longest common subsequence, useful for evaluating structure and flow.  

### Qualitative Methods

Human evaluation with a scoring rubric based on:

- **Accuracy**: Are the steps correct?  
- **Completeness**: Any crucial steps missing?  
- **Clarity**: Simple, unambiguous, and user-friendly?  
- **Format Adherence**: Consistent numbered list output?  

### Benchmark Setup

We will compare three models:

1. **Baseline**: Mistral-7B-Instruct-v0.2 without fine-tuning.  
2. **Fine-Tuned Model (ours)**: Trained with QLoRA.  
3. **Human Reference (Gold Standard)**: Ground truth answers in the test set.  

A held-out test set (15% of data) will be used for final evaluation. Both automatic metrics and **blind human evaluation** will be conducted to minimize bias.

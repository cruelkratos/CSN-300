# CSN-300 Lab Based Project: Story Generation with Fine-Tuned LLMs and Custom Transformers

## Overview
This project focuses on text generation using two distinct approaches:
1. **Fine-tuning Meta's Llama-2-7b-chat model** on the Harry Potter book series.
2. **Implementing a transformer-based language model from scratch** (10M parameters) for story generation.

Both models are trained on the `HarryPotter4Books.txt` dataset to generate creative text in the style of the Harry Potter universe.

---

## Models

### 1. Fine-Tuned Llama-2-7b-chat
- **Approach**: Leveraged Hugging Face's `transformers` and PEFT (Parameter-Efficient Fine-Tuning) with LoRA (Low-Rank Adaptation).
- **Key Features**:
  - 4-bit quantization for memory efficiency.
  - LoRA configuration: Rank=8, Alpha=16, applied to `q_proj` and `v_proj` layers.
  - Trained for 2 epochs with mixed-precision (`fp16`) and gradient accumulation.
- **Use Case**: High-quality, coherent long-form story generation.

### 2. Custom Transformer Model (From Scratch)
- **Architecture**: GPT-style model with:
  - **6 transformer layers**, **6 attention heads**, and **384-dimensional embeddings**.
  - Context window: 256 tokens.
  - 10 million parameters.
- **Training**: 
  - Optimized using AdamW (`lr=3e-4`) for 10,000 iterations.
  - Batch size: 64, dropout: 0.2.
- **Use Case**: Lightweight, interpretable model for educational purposes.

---

## Dataset
- **Source**: `HarryPotter4Books.txt` (4 combined books from the series).
- **Preprocessing**:
  - Split into chunks of 512 tokens for Llama-2.
  - Character-level tokenization for the custom transformer.

---

## Dependencies
```bash
# For Llama-2 fine-tuning
pip install transformers datasets accelerate bitsandbytes peft sentencepiece huggingface_hub

# For custom transformer
pip install torch
```

## Usage

- Fine-Tuned Llama-2

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./llama2-hp-finetuned")
tokenizer = AutoTokenizer.from_pretrained("./llama2-hp-finetuned")

def generate_story(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=500, temperature=0.8)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generate_story("Hogwarts winter night mystery"))
```

- Custom Transformer

```python
# Load saved model checkpoint
model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load("gpt_storygen.pth"))

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
```

## Info
| Model                | Params          | Sample Output Quality       |
|----------------------|-----------------|-----------------------------|
| Fine-Tuned Llama-2   | 7 billion       | Coherent, creative          |
| Custom Transformer   | 10 million      | Basic, short-term coherence |


## Future Work

- Expand dataset to include all 7 Harry Potter books.
- Experiment with larger LoRA ranks for Llama-2.
- Add beam search to the custom transformer.
- Incorporate Wavenet Model
- Add a Frontend and backend for users]

## Team Members
| S.No. | Name             | Enrollment No. |
| ----- | ---------------- | -------------- |
| 1.    | Garv Sethi       | 22115057       |
| 2.    | Granth Gaud      | 22114035       |

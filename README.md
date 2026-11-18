# Prompting, LoRA / QLoRA Fine-Tuning and BM25 Retrieval with Gemma-3-1B

One small open-source LLM, **Gemma-3-1B-it**, used four ways: prompting, parameter-efficient fine-tuning (**LoRA** and **QLoRA**), retrieval-augmented generation with a **BM25** retriever, and fine-tuning combined with retrieval. The task is extractive question answering: the model should answer with the shortest span copied from the passage, and say `NOT IN CONTEXT` when the passage does not contain the answer.

CS 554 *Natural Language Processing*, Worcester Polytechnic Institute, Fall 2026: Assignment 1. Author: Syeda Tamzida Akter.

📓 **[Open the notebook](gemma3_prompting_lora_qlora_bm25_rag.ipynb)** (all outputs included)

---

## 1. Prompting

Compared three input forms for the same questions: the question alone, the question with the passage, and an instruction prompt with the passage. Looked at how the model's use of the passage, the answer length, and its handling of unanswerable questions change. Gemma's chat template is used throughout.

## 2. LoRA vs. QLoRA fine-tuning (SQuAD v2)

Fine-tuned on SQuAD v2 to copy short answer spans and to abstain on unanswerable questions. Only the answer tokens are supervised. Evaluation uses 50 answerable and 50 unanswerable questions with greedy decoding.

| Model | EM | F1 | Abstain % (unanswerable) | Trainable params | Train time | Peak VRAM |
|---|---|---|---|---|---|---|
| Base Gemma-3-1B | 64.0 | 79.4 | 24.0 | – | – | – |
| **LoRA r=8** | **74.0** | **80.1** | **74.0** | 6.5M (0.65%) | 9.5 min | 9.8 GB |
| QLoRA r=8 (4-bit base) | 66.0 | 73.8 | 74.0 | 6.5M | 13.0 min | 9.0 GB |

Fine-tuning taught the model to abstain correctly on unanswerable questions: correct abstentions rose from 24% to 74%. LoRA also improved exact match by 10 points. QLoRA abstained just as well but lost some accuracy because of the 4-bit base model.

## 3. BM25 retrieval (SciQ, 150 hard questions, 12,241 paragraphs)

| Setting | Recall@k | EM | F1 |
|---|---|---|---|
| Closed-book | – | 0.7 | 3.2 |
| **BM25 top-1** | 77.3 | **80.7** | **82.7** |
| BM25 top-3 | 91.3 | 69.3 | 75.8 |
| BM25 top-5 | 93.3 | 58.7 | 69.9 |
| Oracle passage | 100 | 93.3 | 94.6 |

More retrieved passages raise recall but *lower* accuracy: the 1B model is distracted by the extra paragraphs, so top-1 retrieval works best.

## 4. Fine-tuning × retrieval (2 × 2)

Exact match on the same 150 SciQ questions:

| | No RAG | RAG (BM25 top-1) |
|---|---|---|
| Base | 0.7 | 80.7 |
| LoRA fine-tuned | 3.3 | 79.3 |

Retrieval supplies the knowledge (the input **x**): it lifts EM by about 80 points. Fine-tuning changes the model's behavior (its weights **θ**): what to output and when to abstain. It does not add new facts.

## Running it

Runs on Google Colab with a free **T4 GPU**:

1. Accept the license for [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) on Hugging Face, and add a read token to Colab Secrets as `HF_TOKEN`.
2. Run the install cell once, restart the session, then run the rest of the notebook in order.
3. Task 3 needs the course-provided `rag_hard_set_sciq.json` file, which is not included here. SQuAD v2 downloads automatically.

**Stack:** PyTorch · Hugging Face Transformers · PEFT (LoRA / QLoRA) · bitsandbytes · Datasets · rank-bm25

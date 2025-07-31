*# Vietnamese ASR Fine-Tuning with Wav2Vec2*

This repository provides a complete setup to fine-tune a pre-trained Wav2Vec2 model for Vietnamese automatic speech recognition (ASR) using the [Common Voice 17.0](https://commonvoice.mozilla.org/) dataset.

---

## 🚀 Project Overview

Fine-tune the `nguyenvulebinh/wav2vec2-base-vi-vlsp2020` model on Vietnamese speech data from Common Voice. The pipeline:

- Clean and tokenize transcriptions
- Build custom vocabulary (with `|`, `[UNK]`, `[PAD]`)
- Resample audio to 16kHz
- Use `Wav2Vec2Processor` and `Wav2Vec2ForCTC`
- Compute CTC loss with a custom collator
- Evaluate with Word Error Rate (WER)

---

## ✨ Features

- Data cleaning with regex to remove special characters.
- Automatic extraction of unique characters for vocabulary.
- Integration with Hugging Face `datasets` and `transformers` libraries.
- Custom CTC padding and WER computation.
- Training with gradient checkpointing & mixed precision (fp16).

---

## 🛠️ Requirements

- Python 3.8+
- CUDA-enabled GPU (recommended)

Required Python packages are listed in [`requirements-vi.txt`](requirements-vi.txt):

```text
datasets<4.0.0
transformers>=4.31.0
torchaudio
jiwer
accelerate
pyctcdecode>=0.5.0
git+https://github.com/kpu/kenlm.git
```

---

## ⚙️ Installation

1. Clone the repo:

```bash
git clone https://github.com/kienngoxuan/vietnamese-asr-finetune.git
cd vietnamese-asr-finetune
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\activate   # Windows
pip install -r requirements-vi.txt
```

---

## 📂 Dataset Preparation

The notebook/script uses the Hugging Face `datasets` library to load:

- **Training + Validation:** `mozilla-foundation/common_voice_17_0`, `vi`, split=`train+validation`
- **Test:** same dataset, split=`test`

Extra columns (`accent`, `age`, `client_id`, etc.) are removed for simplicity.

---

## 🧹 Data Preprocessing

1. **Inspect random samples:** `show_random_elements()` displays transcriptions.
2. **Clean text:** remove punctuation, quotes, percent signs, etc., and convert to lowercase.
3. **Build vocabulary:** extract unique chars across all sentences.
4. **Add special tokens:**
   - `|` for space
   - `[UNK]` for unknown
   - `[PAD]` for padding

Vocabulary is saved as `vocab.json`.

---

## 🤖 Model and Processor

- **Processor:** `Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vi-vlsp2020")`
- **Model:** custom wrapper from `model_handling.py` in the same repo, loaded with:

```python
model_loader = SourceFileLoader("model_handling", model_script).load_module()
model = model_loader.Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
```

---

## ⚡ Resampling & Feature Extraction

- Audio is cast to 16 kHz via `datasets.cast_column()`.
- `prepare_dataset()` uses `processor(array, sampling_rate=sr)` to extract `input_values`.
- Tokenize text transcriptions to `labels`.

---

## 🛡️ Data Collator for CTC

A custom collator (`DataCollatorCTCWithPadding`) handles:

- Padding audio features (`input_values`).
- Padding labels, replacing padded token IDs with `-100` for CTC loss.

---

## 📏 Word Error Rate (WER)

- Low‑level Levenshtein-based WER implemented in `WordErrorRate`.
- Wrapper `WERMetric` mimics 🤗datasets `wer` metric API.
- `compute_metrics()` decodes predictions & references, then computes WER.

---

## 🎯 Training

Training arguments (via `TrainingArguments`):

```python
training_args = TrainingArguments(
  output_dir="vietnamese-model-test",
  num_train_epochs=10,
  per_device_train_batch_size=1,
  gradient_accumulation_steps=1,
  gradient_checkpointing=True,
  eval_strategy="steps",
  eval_steps=1000,
  logging_steps=1,
  fp16=True,
  learning_rate=5e-5,
  group_by_length=True,
  push_to_hub=False,
)
```

Instantiate and train using 🤗`Trainer`:

```python
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor,
)
trainer.evaluate()
trainer.predict(common_voice_test)
trainer.train()
```

---

## 📊 Evaluation

- Initial evaluation before training: `trainer.evaluate()`.
- Test set metrics: `trainer.predict()` returns WER on test split.

---

## 🚀 Usage

Launch the notebook or run the script:

```bash
jupyter notebook fine_tune_vi_asr.ipynb
# or
python fine_tune_vi_asr.py
```

Monitor training logs and evaluation metrics. Adjust hyperparameters in `TrainingArguments` as needed.

---

## 🗂️ Repository Structure

```
├── requirements-vi.txt    # Dependencies
├── vocab.json             # Generated vocabulary
├── model_handling.py      # Helper for loading CTC model
├── fine_tune_vi_asr.ipynb # Notebook with full pipeline
├── scripts/
│   └── train.py           # Optional script entrypoint
└── README.md              # This file
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to fork and submit pull requests.

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.


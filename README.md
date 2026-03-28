# Wikipedia Content Fetching and Text Summarization System

A lightweight end-to-end NLP pipeline that fetches Wikipedia articles and generates abstractive summaries using Facebook's BART model. The system supports both zero-shot and fine-tuned inference, evaluates summary quality with multiple metrics, and exposes everything through an interactive Streamlit web interface.

---

## Features

- **Wikipedia Integration** — Fetch and preprocess articles by topic via the Wikipedia API
- **Abstractive Summarization** — Generate high-quality summaries with `facebook/bart-large-cnn`
- **Fine-tuning Support** — Train a custom BART model on your own Wikipedia corpus
- **Quality Evaluation** — Score summaries using ROUGE-1/2/L, paragraph coverage, and information density
- **Model Comparison** — Side-by-side benchmarking of pre-trained vs. fine-tuned models
- **Web Interface** — Clean Streamlit UI with real-time summary generation and metric display
- **GPU Acceleration** — Automatic CUDA detection and utilisation

---

## Project Structure

```
├── light_app.py            # Streamlit web application
├── text_summarizer.py      # Summarization and evaluation logic
├── wiki_data_collector.py  # Wikipedia article fetching and storage
├── train_model.py          # BART fine-tuning pipeline
├── model_comparison.py     # Pre-trained vs. fine-tuned benchmarking
├── wikipedia_utils.py      # Wikipedia helper utilities
├── qa_engine.py            # Question-answering engine
├── question_gen.py         # Question generation utilities
└── requirement.txt         # Python dependencies
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| [Streamlit](https://streamlit.io/) | Web application framework |
| [PyTorch](https://pytorch.org/) | Deep learning backend |
| [Transformers](https://huggingface.co/docs/transformers) | BART model and tokenizer |
| [wikipedia](https://pypi.org/project/wikipedia/) | Wikipedia content fetching |
| [rouge-score](https://pypi.org/project/rouge-score/) | Summary evaluation metrics |
| [NLTK](https://www.nltk.org/) | Text preprocessing |
| [scikit-learn](https://scikit-learn.org/) | Supporting ML utilities |

---

## Installation

**Prerequisites:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/your-username/Wikipedia-content-fetching-and-text-summarization-system.git
cd Wikipedia-content-fetching-and-text-summarization-system

# Install dependencies
pip install -r requirement.txt
```

> The first run will automatically download the `facebook/bart-large-cnn` model weights (~1.6 GB). A stable internet connection is required.

---

## Usage

### 1. Run the Web App (recommended)

```bash
streamlit run light_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser, enter a Wikipedia topic, and receive a summary with quality metrics instantly.

### 2. Collect Training Data (optional)

```bash
python wiki_data_collector.py
```

Fetches random Wikipedia articles (minimum 1,000 words each) and saves them to `data/wiki_training_data.json`.

### 3. Fine-tune the Model (optional)

```bash
python train_model.py
```

Fine-tunes `facebook/bart-large-cnn` on the collected corpus. The trained model is saved to `models/wiki_summarizer/`. GPU is strongly recommended.

### 4. Compare Models (optional)

```bash
python model_comparison.py
```

Runs a benchmark across a configurable number of test articles and outputs a side-by-side metric report for the pre-trained and fine-tuned models.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| ROUGE-1 | Unigram overlap between generated and reference summary |
| ROUGE-2 | Bigram overlap |
| ROUGE-L | Longest common subsequence |
| Paragraph Coverage | Fraction of source paragraphs reflected in the summary |
| Information Density | Ratio of unique informative tokens to total summary tokens |
| Comprehensive Score | Weighted aggregate of all the above metrics |

---

## Notes

- Model training is resource-intensive; a CUDA-capable GPU with at least 8 GB VRAM is recommended.
- Data collection makes live requests to the Wikipedia API; run in batches to avoid rate limits.
- Summary length is configurable via the `max_length` and `min_length` parameters in `text_summarizer.py`.

---

## License

This project is licensed under the [MIT License](LICENSE).

# Universal Dependencies POS Tagging with LLMs

[![NLP with LLMs](https://img.shields.io/badge/NLP-LLMs-blue)](https://universaldependencies.org/)
[![Python](https://img.shields.io/badge/python-3.8%2B-green)](https://www.python.org/)
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini%202.0-orange)](https://ai.google.dev/)

## Project Overview

This repository contains code and analysis for fine-grained Part-of-Speech (POS) tagging using Large Language Models (LLMs), specifically focused on the Universal Dependencies (UD) framework. The project explores how LLMs can perform complex linguistic annotation tasks and addresses the challenges of tokenization and POS tagging in a unified pipeline.

## Key Features

- **Universal Dependencies POS tagging** with Google Gemini 2.0 Flash Lite
- **Segmentation pipeline** that follows UD tokenization guidelines
- **Comprehensive error analysis** comparing LLM performance against traditional methods
- **Evaluation framework** for both tokenization and tagging accuracy
- **Integrated pipeline** that handles both segmentation and tagging in one flow

## Installation and Setup

### Prerequisites
* Python (> 3.11)
* Git
* uv (https://docs.astral.sh/uv/getting-started/)
* Visual Studio Code

### Environment Setup

1. Create a folder for the assignment: 
   ```bash
   mkdir hw1; cd hw1
   ```
2. Retrieve the dataset we will use and the code from this repo:
   ```bash
   git clone https://github.com/UniversalDependencies/UD_English-EWT.git
   git clone https://github.com/melhadad/nlp-with-llms-2025-hw1.git
   ```
3. Load the required python libraries:
   ```bash
   cd nlp-with-llms-2025-hw1; uv sync
   ```
4. Define your API keys in either gemini_key.ini or grok_key.ini
   ```bash
   # Unix like
   source grok_key.ini
   export GROK_API_KEY=$GROK_API_KEY
   ```
   
   For Google Gemini:
   ```bash
   export GOOGLE_API_KEY="your-api-key"  # On Windows: set GOOGLE_API_KEY=your-api-key
   ```
5. Activate the project virtual env: 
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
6. Open ud_pos_tagger_sklearn.ipynb in VS Code and verify you can execute the cells.

## Dataset

The project uses the [Universal Dependencies English-EWT](https://universaldependencies.org/treebanks/en_ewt/index.html) dataset. The code expects the dataset directory structure as follows:

```
UD_English-EWT/
├── en_ewt-ud-dev.conllu
├── en_ewt-ud-test.conllu
└── en_ewt-ud-train.conllu
```

## Running the Code

### Basic POS Tagging

```bash
# Run the basic POS tagger
uv run ud_pos_tagger_gemini.py
```

### Tokenization and Segmentation

```bash
# Run the segmentation model
uv run ud_pos_llm_segmentor_gemini.py
```

### Integrated Pipeline

```bash
# Run the improved LLM tagger with integrated segmentation
uv run ud_pos_improved_llm_tagger.py
```

### Analysis Notebook

To analyze the results and view the visualizations:

```bash
jupyter notebook ud_pos_tagger_gemini.ipynb
```

## Project Structure

- **`ud_pos_tagger_gemini.py`**: Main POS tagging implementation using Gemini
- **`ud_pos_llm_segmentor_gemini.py`**: Specialized tokenization model for UD guidelines
- **`ud_pos_improved_llm_tagger.py`**: Integrated pipeline for segmentation and tagging
- **`utils.py`**: Helper functions for data processing and evaluation
- **`schema.py`**: Data structures and type definitions
- **`prompts.py`**: LLM prompts for tagging and segmentation
- **`ud_pos_tagger_gemini.ipynb`**: Jupyter notebook with detailed analysis and visualizations

## Results and Findings

### POS Tagging Performance

The LLM tagger achieves strong performance on Universal Dependencies POS tagging, with the following key findings:

1. **Strong overall accuracy**, with particular strengths in:
   - Adpositions (ADP)
   - Proper nouns (PROPN) 
   - Common nouns (NOUN)
   - Verbs (VERB)

2. **Improvement areas** compared to traditional machine learning approaches:
   - Pronouns (PRON) recognition
   - Distinguishing determiners (DET) from pronouns
   - Particle (PART) vs. adverb (ADV) disambiguation

### Segmentation Impact

Our analysis of tokenization shows:

- 38.7% average error reduction when using proper UD tokenization
- Most significant improvements on sentences with hyphenated compounds and punctuation
- Most challenging segmentation cases: hyphenated terms, contractions, and special punctuation

### Challenging Cases

The LLM tagger struggles most with:

1. Deictic words that can be pronoun or determiner - "this/that/these/those"
2. Discourse pronoun "there" vs. locative adverb
3. Subordinating conjunction vs. adposition - words like "for", "in", "to"
4. Verb-particle/adverb vs. preposition - "up", "out", "in", "off", "on"
5. Possessive pronouns classified as determiners

## Future Work

- Experiment with fine-tuning approaches for the LLM
- Explore parameter-efficient adaptation for specialized linguistic domains
- Implement additional languages from Universal Dependencies
- Create a more robust evaluation framework for cross-linguistic performance
- Develop a web interface for interactive POS tagging demonstrations

## Acknowledgments

- Universal Dependencies project for the dataset and guidelines
- Google for providing access to the Gemini API
- The NLP community for benchmarks and evaluation methodologies
- BGU CS Course 'NLP with LLMs' - Spring 2025 - Michael Elhadad

Authors: Gil Barel and Daniel Ohayon

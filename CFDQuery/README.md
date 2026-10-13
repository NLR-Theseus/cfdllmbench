# LLM Computational Fluid Dynamics (CFD) Benchmark

A comprehensive framework for evaluating the performance of various Large Language Models (LLMs) on computational fluid dynamics multiple-choice questions.

## Description

This project provides an end-to-end solution for assessing how well different LLMs (from OpenAI, Anthropic, Google, and open-source models) perform on CFD-specific knowledge tests. It automates the process of sending multiple-choice questions to various models, parsing their responses, and generating detailed performance reports.

## Features

- **Multi-Model Support**: Evaluates performance across various LLM providers:
  - OpenAI models (GPT-4o, O3-mini)
  - Anthropic models (Claude 3.5 Sonnet, Claude 3.5 Haiku)
  - Google models (Gemini 2.0 Flash)
  - Open-source models (Llama 3.2, Gemma 2)
- **Robust Response Parsing**: Includes both strict and tolerant parsing mechanisms to handle different response formats
- **Retry Logic**: Attempts up to 3 retries when response parsing fails
- **Comprehensive Reporting**:
  - Overall evaluation summaries
  - Per-model complete answers (correct and incorrect)
  - Detailed analysis of incorrect answers
  - Performance metrics and accuracy statistics

## Prerequisites

- Python 3.8+
- API keys for the following services:
  - OpenAI API
  - Anthropic API
  - Google AI Studio API
- Ollama installation for local model inference

## Installation

1. Install the required dependencies:
```
pip install openai anthropic google-generativeai pydantic langchain-ollama
```

2. Set up environment variables for API keys:
```
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export GOOGLE_API_KEY="your_google_key"
```

## Usage

1. Prepare your CFD questions in the required JSON format:

```json
{
  "CFD QA": [
    {
      "question_index": 1,
      "question_content": "What is the Navier-Stokes equation primarily used for?",
      "options": [
        {"option_index": 1, "option_content": "Calculating lift forces"},
        {"option_index": 2, "option_content": "Describing fluid flow behavior"},
        {"option_index": 3, "option_content": "Computing structural mechanics"},
        {"option_index": 4, "option_content": "Optimizing heat transfer"}
      ],
      "correct_option_index": 2
    },
    // More questions...
  ]
}
```

2. Run the evaluation:

```
python llms.py
```

3. Review the results in the generated output directories:
   - `model_evaluation_results_[TIMESTAMP].json`: Summary of all model performances
   - `wrong_questions_[TIMESTAMP]/`: Directory containing incorrect answers for each model
   - `complete_results_[TIMESTAMP]/`: Directory containing all responses from each model

## Project Structure

```
llm-cfd-evaluation/
│
├── llms.py       # Main evaluation script
|
│
├── results/                      # Generated after running the evaluation
│   ├── model_evaluation_results_*.json
│   ├── wrong_questions_*/
│   └── complete_results_*/
│
└── README.md
```

## Dataset
The latest (V2) dataset is included in this folder: [`CFDQuery.json`](CFDQuery.json).

The dataset used in the paper can be downloaded from `https://www.kaggle.com/datasets/nithinsekhar/cfdquery/data`.
- Download the file named 'CFDQuery.json' from the dataset to the same folder as llms.py

## Acknowledgments

- CFD question dataset contributors
- The API providers: OpenAI, Anthropic, Google
- The Ollama project for local model inference

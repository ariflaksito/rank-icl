# Rank-ICL: Ranking-based In-context Learning for Search Result Explanation

This repository provides code to generate **query–document relevance explanations** using large language models (LLMs), supporting **zero-shot**, **few-shot**, and **Rank-ICL** settings.

![rank-icl framework](Rank-ICL.png)

---
### Abstract

Explanations in search results typically consist of text snippets or short passages presented alongside retrieved documents to help users efficiently assess relevance. 
While large language models (LLMs) have demonstrated strong performance across a wide range of language understanding and generation tasks, prior work on their use to generate explanations in search results remains relatively sparse. In this study, we investigate the use of decoder-only LLMs to generate explanations in the search results. To improve explanation quality in low-supervision settings, we introduce a ranking-based strategy for selecting informative few-shot examples in in-context learning. 
Rather than relying on randomly chosen demonstrations, relevant examples are dynamically retrieved based on retrieval functions to the input query–document pair.
Evaluation on WikiSA and ExaRank shows that ranking-based few-shot prompting generally improves over zero-shot prompting and achieves competitive performance against random-shot prompting. However, its effectiveness varies across datasets, indicating that retrieval-based demonstration selection is beneficial but not uniformly superior in all settings.
### Resluts
![Results Table](images/results_table.png)
## Models

We evaluate three decoder-only LLMs: **[LLaMA 3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)**, **[Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B-Instruct)**, and **[Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)**.

Best decoding settings per model and dataset (used consistently across zero-shot, random-shot, and Rank-ICL):

<table align="center">
  <thead>
    <tr>
      <th>Model</th>
      <th>Dataset</th>
      <th>temp</th>
      <th>top-p</th>
      <th>top-k</th>
      <th>penalty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">LLaMA 3.1-8B</a></td>
      <td>WikiSA</td>
      <td>0.2</td><td>0.9</td><td>10</td><td>1.1</td>
    </tr>
    <tr>
      <td>ExaRank</td>
      <td>0.1</td><td>0.8</td><td>0</td><td>1.0</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2-7B-Instruct">Qwen2-7B</a></td>
      <td>WikiSA</td>
      <td>0.2</td><td>0.9</td><td>10</td><td>1.0</td>
    </tr>
    <tr>
      <td>ExaRank</td>
      <td>0.1</td><td>0.8</td><td>10</td><td>1.0</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3">Mistral-7B</a></td>
      <td>WikiSA</td>
      <td>0.1</td><td>0.8</td><td>10</td><td>1.0</td>
    </tr>
    <tr>
      <td>ExaRank</td>
      <td>0.1</td><td>0.8</td><td>10</td><td>1.0</td>
    </tr>
  </tbody>
</table>

## Prompts

### WikiSA

**Zero-shot**
```
You are an AI assistant that explain how query and document are related.
Provide a few words of aspect explanation max 5 words.
query: {query}.
document: {doc}.
aspect:
```

**Few-shot / Rank-ICL**
```
You are an AI assistant that explain how query and document are related.
I give you n examples below:
Example 1
query: {query}.
document: {doc}.
aspect: {explanation}.
...
Example n
query: {query}.
document: {doc}.
aspect: {explanation}.
Provide a few words of aspect explanation max 5 words.
query: {query}.
document: {doc}.
aspect:
```

---

### ExaRank

**Zero-shot**
```
You are an AI assistant that explain how query and document are related.
Provide an explanation if the query is related to the document.
query: {query}.
document: {doc}.
explanation:
```

**Few-shot / Rank-ICL**
```
You are an AI assistant that explain how query and document are related.
I give you n examples below:
Example 1
query: {query}.
document: {doc}.
explanation: {explanation}.
...
Example n
query: {query}.
document: {doc}.
explanation: {explanation}.
Provide an explanation if the query is related to the document.
query: {query}.
document: {doc}.
explanation:
```



<!--add poster later-->

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Run
```bash
python -m run_generate.py [ARGS]
```

### 3. Example
```bash
python -m run_generate.py \
  --hf-token YOUR_TOKEN \
  --dataset wiki \
  --test-size 2000 \
  --samples 3 \
  --function sbert \
  --prompt rag \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --top-p 0.8 --top-k 10 --temp 0.1 \
  --max-tokens 100 --penalty 1.0 \
  --output outputs/wiki_llama_sbert.csv \
  --metrics-log logs/wiki_llama_sbert.csv \
```

## Dataset
- Code to construct the WikiSA dataset is available at the directory `data/wikisa/`
- The ExaRank dataset is available at the directory `data/exarank1/`

## Output
Each run generates:
- CSV output (--output) with reference and generated explanations
- Metrics log (--metrics-log) evaluate using ROUGE-1, METEOR, BERTScore metrics

<!-- ## Citation
```bibtex
@inproceedings{rank-icl2026,
  title={...},
  author={...},
  year={2026}
}
``` -->

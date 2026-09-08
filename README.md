# LLM Evaluation CI/CD Pipeline

Automated regression testing and observability for LLM and RAG applications.

The project evaluates a candidate LLM or RAG pipeline whenever prompts, models, application logic, or knowledge-base content change. It measures response quality, latency, and cost, records evaluation runs in Langfuse, and exposes a CI status that can be used as a required merge check.

## Architecture

```text
                 GitHub Repository
        ┌─────────────────────────────┐
        │ prompts/                    │
        │ knowledge_base/             │
        │ config/model.yaml           │
        │ config/evaluation.yaml      │
        │ data/golden_dataset.jsonl   │
        └──────────────┬──────────────┘
                       │
                 Pull Request / Push
                       │
                       ▼
               GitHub Actions CI
                       │
                       ▼
              Evaluation Runner
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
     Candidate RAG             Golden Dataset
       Pipeline                 100+ cases
          │                         │
          └────────────┬────────────┘
                       ▼
                  Evaluators
          ┌────────────┼─────────────┐
          │            │             │
     Relevancy    Faithfulness   Correctness
          │
     Hallucination
                       │
                       ▼
                   Langfuse
        ┌────────────────────────────┐
        │ Traces                     │
        │ Experiments                │
        │ Evaluator scores           │
        │ Latency / tokens / cost    │
        │ Historical comparisons     │
        │ Dashboards                 │
        └──────────────┬─────────────┘
                       │
                       ▼
                   CI Gate
                       │
            ┌──────────┴─────────┐
            ▼                    ▼
           PASS                 FAIL
            │                    │
          Merge              Block merge
```

## Responsibilities

| Component | Responsibility |
|---|---|
| Git | Source of truth for prompts, model configuration, evaluation thresholds, golden dataset, and knowledge-base content |
| LLM/RAG application | System under test |
| Langfuse | Traces, evaluation experiments, feedback scores, latency, token usage, cost, historical comparisons, and dashboards |
| GitHub Actions | Automated evaluation execution on pull requests and pushes |
| Evaluation runner | Executes the candidate pipeline against the golden dataset |
| Evaluators | Measure correctness, answer relevancy, faithfulness, hallucination behavior, and abstention |
| CI gate | Applies deterministic quality, latency, and cost thresholds |
| GitHub branch protection | Prevents merges when the evaluation workflow fails |

## Evaluation Metrics

### Quality

- Answer relevancy
- Correctness
- Faithfulness to retrieved sources
- Hallucination rate
- Abstention accuracy

### Performance

- p50 latency
- p95 latency
- Error rate

### Cost

- Input tokens
- Output tokens
- Mean cost per query
- Total evaluation cost

## Golden Dataset

The golden dataset is version controlled in Git and synchronized with Langfuse for experiment execution, scoring, and analysis.

```text
data/golden_dataset.jsonl
```

Example:

```json
{
  "id": "refund_001",
  "inputs": {
    "question": "Can I return an item after 30 days?"
  },
  "reference_outputs": {
    "answer": "Items must be returned within 30 days."
  },
  "metadata": {
    "category": "refund",
    "difficulty": "normal",
    "should_abstain": false
  }
}
```

The dataset should contain representative production scenarios, paraphrases, boundary cases, ambiguous questions, unanswerable questions, conflicting-context cases, retrieval edge cases, and adversarial inputs.

## CI Quality Gates

Evaluation thresholds are version controlled.

Example:

```yaml
gates:
  max_hallucination_rate: 0.05

  latency:
    p95_max_ms: 5000
    max_regression_percent: 20

  quality:
    min_answer_relevancy: 0.80
    min_faithfulness: 0.85

  cost:
    max_mean_cost_per_query_usd: 0.02
```

The evaluation process exits with a non-zero status when a blocking threshold is violated. GitHub can then use the workflow as a required status check.

## Langfuse Observability and Evaluation

Langfuse acts as the evaluation and observability backend for traces, experiments, scores, latency, token usage, cost, and historical comparisons.

Each evaluation run should record metadata such as:

```text
git_sha
branch
pull_request
model
prompt_hash
knowledge_base_hash
dataset_version
```

This allows experiments to be compared across model, prompt, retrieval, and knowledge-base changes.

Relevant Langfuse data includes:

- End-to-end request traces
- Retrieval traces
- LLM calls
- Inputs and outputs
- Evaluator feedback
- Latency
- Token usage
- Cost
- Experiment history
- Baseline and candidate comparisons


## Technology Stack

- Python
- GitHub Actions
- Langfuse
- LLM provider SDK or LiteLLM
- Custom evaluators or LLM-as-a-judge evaluators
- pytest
- Vector store or retriever for the RAG implementation

LangChain or LangGraph can be used when appropriate, but Langfuse is framework-agnostic and can instrument applications implemented with other LLM stacks.

## Environment Configuration

Example:

```bash
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com


OPENAI_API_KEY=
```

Secrets used by CI should be configured through GitHub Actions secrets and must not be committed to the repository.

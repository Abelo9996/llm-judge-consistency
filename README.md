# LLM-as-Judge Consistency: How Reliable Is Your Evaluator?

**Paper**: "How Reliable Is Your LLM Judge? Measuring Scoring Consistency and Bias in LLM-Based Evaluation"

## Key Question
LLM-as-judge evaluation is now the default for benchmarking (MT-Bench, AlpacaEval, Chatbot Arena). But how **consistent** is the judge itself?

We run the same evaluation N times and measure:
1. Does the judge give the same score each time?
2. Does score variance depend on the model being judged?
3. How do known biases (position, verbosity, self-preference) affect consistency?
4. Do different judge models agree with each other?

## What's New vs Prior Work
- [Zheng et al. (2023)](https://arxiv.org/abs/2306.05685) introduced MT-Bench & LLM-as-judge but only measured inter-judge agreement, not intra-judge consistency
- [Dubois et al. (2024)](https://arxiv.org/abs/2404.04475) studied bias in AlpacaEval but focused on length, not scoring variance
- [Li et al. (2024)](https://arxiv.org/abs/2404.12272) surveyed biases but didn't measure run-to-run consistency
- We provide the **first systematic study of intra-judge scoring consistency** across multiple judges, evaluation formats, and bias conditions

## Experiments

### Experiment 1: Intra-Judge Consistency
- Take 100 response pairs from MT-Bench / AlpacaEval
- Judge each pair N=50 times with the same model
- Measure: score variance, flip rate (how often winner changes), ICC

### Experiment 2: Cross-Judge Agreement
- Same 100 pairs judged by GPT-4o, GPT-4o-mini, Claude Sonnet, Llama 70B
- Measure: inter-judge agreement (Cohen's kappa), systematic disagreement patterns

### Experiment 3: Bias × Consistency Interaction
- **Position bias**: Swap response order, measure score change
- **Verbosity bias**: Same content, different lengths
- **Self-preference**: Does GPT-4o rate GPT-4o outputs higher?
- Key insight: biases may INCREASE consistency (always picks verbose response) or DECREASE it (random noise)

### Experiment 4: Evaluation Format Sensitivity
- Pointwise scoring (1-10) vs pairwise comparison vs reference-based grading
- Which format produces most consistent judgments?

## Metrics
- **Score Variance**: std of scores across repeated evaluations
- **Flip Rate**: % of times the winner changes across repeats
- **Intraclass Correlation (ICC)**: reliability coefficient
- **Position Bias Index**: score delta when swapping A/B order
- **Verbosity Correlation**: Pearson r between response length and score
- **Self-Preference Rate**: win rate of judge's own model vs others
- **Inter-Judge Kappa**: Cohen's kappa between judge pairs

## Structure
```
llm-judge-consistency/
├── src/
│   ├── evaluators/     # Judge implementations (pointwise, pairwise, reference)
│   ├── metrics/        # Consistency & bias metrics
│   ├── runners/        # Experiment orchestration
│   └── analysis/       # Results analysis & figures
├── data/               # Evaluation datasets (MT-Bench, AlpacaEval samples)
├── configs/            # Experiment configs
├── results/            # Raw results
├── figures/            # Generated figures
├── paper/              # LaTeX source
└── README.md
```

## Quick Start
```bash
pip install -r requirements.txt
cp .env.example .env  # Add API keys
python -m src.runners.run_experiment --config configs/main.yaml
python -m src.analysis.generate_figures
```

## Models (as judges)
- GPT-4o / GPT-4o-mini (OpenAI)
- Claude Sonnet 4.5 (Anthropic) — if available
- Llama 3.1 70B (Together) — if available

## License
MIT

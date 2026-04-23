# The Coin Flip Judge: Quantifying Inconsistency and Position Bias in LLM-as-a-Judge Evaluation

**Paper**: ["The Coin Flip Judge: Quantifying Inconsistency and Position Bias in LLM-as-a-Judge Evaluation"](paper/main.pdf)  
**Author**: Abel Yagubyan (Independent Researcher)  
**arXiv**: https://github.com/Abelo9996/llm-judge-consistency

## Overview

LLM-as-a-Judge is now the default evaluation paradigm for benchmarking language models (MT-Bench, AlpacaEval, Chatbot Arena). But how **consistent** is the judge itself when re-run on the same input?

We run 8,700+ pairwise and pointwise evaluations (50 trials per question) and find:

- **14% mean flip rate** — one in seven pairwise comparisons changes if re-run
- **72% position bias** in GPT-4o-mini: the first-presented response wins majority preference (*p* = 0.024, sign test)
- **Pairwise-pointwise paradox**: judges confidently pick winners even when pointwise scores don't differ (*p* > 0.1)
- **κ = 0.51** cross-judge agreement — choosing the wrong judge model flips ~25% of outcomes
- **t=0 is necessary but not sufficient**: flip rates drop 43–79% but don't reach zero
- **25% prompt sensitivity**: semantically equivalent prompts with different wording change majority outcomes in 1 in 4 cases

## What's New vs Prior Work

| Study | Contribution | Our difference |
|---|---|---|
| Zheng et al. (2023) MT-Bench | Introduced LLM-as-Judge; measured inter-model agreement | We measure intra-judge consistency across 50 repeated trials |
| Wang et al. (2023) FairEval | Systematic bias survey (position, verbosity) | We add pairwise-pointwise paradox + temperature + prompt ablations |
| Stureborg et al. (2024) | Broad inconsistency characterization | We focus on 50-trial flip rates + paradox on a single cohesive dataset |
| Dubois et al. (2024) AlpacaFarm | Length bias in instruction following | Different bias type; we focus on run-to-run variance |

## Experiments

### Experiment 1: Intra-Judge Consistency (main)
- 29 response pairs across 10 categories
- Responses: GPT-4o-mini (A) vs GPT-4o (B)
- Judges: GPT-4o-mini, GPT-4.1-mini
- 50 pairwise + 50×2 pointwise trials per (judge, question) = **8,700 API calls**
- Metrics: flip rate, position bias index, pairwise-pointwise gap, cross-judge κ

### Experiment 2: Temperature Ablation
- Same 29 questions, t=0 (deterministic decoding)
- 10 trials per (judge, question) = **580 API calls**

### Experiment 3: Prompt Template Sensitivity
- 10 questions (one per category), 2 prompt templates × 20 trials
- Both judges = **800 API calls**

## Key Results

| Metric | GPT-4o-mini | GPT-4.1-mini |
|---|---|---|
| Mean flip rate (t=1) | 13.3% | 13.9% |
| Max flip rate | 46% | 56% |
| Position bias (A wins) | 72%* | 59% |
| Mean pointwise gap | 0.19/10 | 0.36/10 |
| Flip rate (t=0) | 2.8% | 7.9% |

\* *p* = 0.024, sign test. Cross-judge κ = 0.51, agreement = 76%.

## Repository Structure

```
llm-judge-consistency/
├── src/
│   ├── evaluators/judges.py      # Pairwise + pointwise judge implementations
│   ├── metrics/consistency.py    # Flip rate, PBI, ICC, kappa
│   └── runners/run_experiment.py # Experiment orchestration
├── data/eval_pairs.json          # 29 evaluation question-response pairs
├── configs/main.yaml             # Experiment configuration
├── results/                      # Raw JSON results for all 3 experiments
├── paper/
│   ├── main.tex                  # LaTeX source
│   ├── main.pdf                  # Compiled paper
│   ├── figures/                  # All 7 figures (PDF + PNG)
│   └── references.bib
├── gen_figures.py                # Regenerate figs 1–5
├── gen_fig6.py                   # Temperature ablation figure
├── gen_fig7.py                   # Prompt sensitivity figure
└── requirements.txt
```

## Reproducing Figures

```bash
pip install -r requirements.txt
python gen_figures.py   # Figs 1–5
python gen_fig6.py      # Fig 6: temperature ablation
python gen_fig7.py      # Fig 7: prompt sensitivity
```

## Limitations & Future Work

- **Single provider**: Both judges are OpenAI models. Extending to Llama, Claude, Gemini is the primary next step.
- **29 questions**: Category-level comparisons lack statistical power; scaling to 100 questions would enable significance testing.
- **Self-preference confound**: GPT-4o-mini judges GPT-4o-mini (Response A) vs GPT-4o (Response B); position bias and self-preference are not fully disentangled.
- **Temperature comparison is approximate**: 50 trials at t=1 vs 10 trials at t=0 have different confidence intervals.

## Citation

```bibtex
@article{yagubyan2026coinflip,
  title={The Coin Flip Judge: Quantifying Inconsistency and Position Bias in {LLM}-as-a-Judge Evaluation},
  author={Yagubyan, Abel},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT

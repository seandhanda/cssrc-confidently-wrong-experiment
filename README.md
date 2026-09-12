# Characterizing Model-Specific Confidence–Accuracy Decoupling Under Controlled Feature Corruption

This repository contains the complete reproducible experiment and generated results for the research paper published in CSSRC 2026.

## Abstract

Classification confidence is useful only when it tracks the probability that a prediction is correct. Although accuracy degradation under distribution shift is expected, a more consequential failure occurs when accuracy falls without a corresponding reduction in confidence. We study this confidence–accuracy decoupling under two controlled forms of tabular feature corruption: additive Gaussian noise and random feature masking. Logistic regression, random forest, and histogram gradient boosting classifiers are evaluated on three standard datasets across 20 paired stratified resamples. We measure accuracy, mean confidence, expected calibration error, Brier score, negative log-likelihood, and selective-prediction risk, with and without temperature scaling fitted on clean calibration data. Under maximum Gaussian noise, logistic regression retains 91.7% mean confidence at 66.4% accuracy, while gradient boosting retains 77.3% confidence at 56.4% accuracy. Random forests respond in the opposite direction, averaging 50.9% confidence at 62.0% accuracy. Diagnostic measurements explain this contrast: the mean decision margin on incorrect logistic-regression predictions rises from 1.05 to 3.90, whereas random-forest tree-vote entropy rises from 0.34 to 0.84. Moreover, clean-fitted temperature scaling improves calibration for seven of nine model–dataset pairs on clean data but only three under maximum Gaussian noise. These results show that confidence failure is model- and corruption-dependent: shift can produce either confident errors or excessive uncertainty, and calibration established in-distribution need not transfer after corruption.

## Research Question

When feature corruption reduces predictive accuracy, does model confidence decline appropriately—or can accuracy and confidence deteriorate at different rates?

## Experimental Design

- **Datasets:** Breast Cancer Wisconsin, Wine, and Digits
- **Models:** Logistic regression, random forest, and histogram gradient boosting
- **Controlled shifts:** Additive Gaussian noise and random feature masking
- **Calibration:** Raw probabilities and temperature scaling fitted on clean calibration data
- **Metrics:** Accuracy, mean confidence, confidence gap, ECE, Brier score, negative log-likelihood, and risk–coverage behavior
- **Evaluation:** 20 paired stratified train/calibration/test splits

## Reproduce the Experiment

Python 3.10 or newer is recommended. No IDE, Jupyter notebook, Miniforge, or Conda installation is required.

```bash
python -m pip install -r requirements.txt
python experiment.py
python diagnostics.py
```

For a faster environment check:

```bash
python experiment.py --quick
python diagnostics.py --quick
```

The scripts use datasets bundled with scikit-learn and do not download external data.

## Repository Structure

```text
.
├── experiment.py       # Main experiment and Figures 1–5
├── diagnostics.py      # Mechanism diagnostics and Figure 6
├── requirements.txt    # Python dependencies
├── results/            # Numerical results and run metadata
└── figures/            # Generated figures in PNG and PDF formats
```

## Outputs

- `results/metrics.csv`: Metrics for every model, split, corruption type, and severity
- `results/reliability.csv`: Calibration-bin statistics
- `results/risk_coverage.csv`: Selective-prediction risk curves
- `results/summary.csv`: Aggregated results and confidence intervals
- `results/diagnostics.csv`: Decision-margin and tree-vote diagnostics
- `results/run_metadata.json`: Runtime, package versions, and experiment settings
- `figures/figure_*.png` and `figures/figure_*.pdf`: Publication-quality figures

## Reproducibility Notes

All preprocessing parameters are fitted using training data only. Temperature scaling is fitted on a separate clean calibration split. Within each resample, all models are evaluated using the same split and corrupted test observations, enabling paired comparisons. Random seeds and package versions are recorded in `results/run_metadata.json`.

## Citation

If you use this repository, please cite the accompanying paper:

> *Characterizing Model-Specific Confidence–Accuracy Decoupling Under Controlled Feature Corruption.* CSSRC 2026.

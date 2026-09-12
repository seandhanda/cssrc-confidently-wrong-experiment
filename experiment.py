"""Run all experiments and generate paper-ready result files and figures."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from scipy.optimize import minimize_scalar
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
EPS = 1e-12

DATASETS = {
    "Breast Cancer": load_breast_cancer,
    "Wine": load_wine,
    "Digits": load_digits,
}

MODEL_NAMES = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
MODEL_COLORS = {
    "Logistic Regression": "#0072B2",
    "Random Forest": "#D55E00",
    "Gradient Boosting": "#009E73",
}


def make_model(name: str, seed: int, quick: bool):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=3000, solver="lbfgs", C=1.0)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=80 if quick else 250,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "Gradient Boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=60 if quick else 150,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )
    raise ValueError(name)


def normalized_probabilities(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, EPS, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(normalized_probabilities(probs)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    probs = normalized_probabilities(probs)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        return log_loss(y, apply_temperature(probs, temperature), labels=np.arange(probs.shape[1]))

    fit = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(fit.x))


def calibration_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 15):
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = (prediction == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (confidence < upper if index < n_bins - 1 else confidence <= upper)
        count = int(mask.sum())
        if count:
            bin_conf = float(confidence[mask].mean())
            bin_acc = float(correct[mask].mean())
            ece += count / len(y) * abs(bin_acc - bin_conf)
            rows.append((index, lower, upper, count, bin_conf, bin_acc))
    return float(ece), rows


def multiclass_brier(probs: np.ndarray, y: np.ndarray) -> float:
    targets = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - targets) ** 2, axis=1)))


def risk_coverage(probs: np.ndarray, y: np.ndarray):
    confidence = probs.max(axis=1)
    errors = (probs.argmax(axis=1) != y).astype(float)
    order = np.argsort(-confidence, kind="stable")
    sorted_errors = errors[order]
    cumulative_risk = np.cumsum(sorted_errors) / np.arange(1, len(y) + 1)
    coverage = np.arange(1, len(y) + 1) / len(y)
    aurc = float(scipy.integrate.trapezoid(cumulative_risk, coverage))
    grid = np.linspace(0.1, 1.0, 10)
    risks = []
    for target_coverage in grid:
        index = max(0, int(np.ceil(target_coverage * len(y))) - 1)
        risks.append((float(target_coverage), float(cumulative_risk[index])))
    return aurc, risks


def corrupt(X: np.ndarray, shift: str, severity: float, seed: int) -> np.ndarray:
    if severity == 0:
        return X.copy()
    rng = np.random.default_rng(seed)
    if shift == "Gaussian noise":
        return X + rng.normal(0.0, severity, size=X.shape)
    if shift == "Feature masking":
        mask = rng.random(X.shape) < severity
        shifted = X.copy()
        shifted[mask] = 0.0
        return shifted
    raise ValueError(shift)


def metric_row(probs: np.ndarray, y: np.ndarray):
    probs = normalized_probabilities(probs)
    prediction = probs.argmax(axis=1)
    accuracy = float(accuracy_score(y, prediction))
    confidence = float(probs.max(axis=1).mean())
    ece, bins = calibration_bins(probs, y)
    aurc, risks = risk_coverage(probs, y)
    return {
        "accuracy": accuracy,
        "mean_confidence": confidence,
        "confidence_gap": confidence - accuracy,
        "ece": ece,
        "brier": multiclass_brier(probs, y),
        "nll": float(log_loss(y, probs, labels=np.arange(probs.shape[1]))),
        "aurc": aurc,
    }, bins, risks


def run_experiments(seeds: int, quick: bool):
    metric_rows, reliability_rows, risk_rows = [], [], []
    gaussian_levels = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    masking_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    total = len(DATASETS) * seeds
    completed = 0

    for dataset_index, (dataset_name, loader) in enumerate(DATASETS.items()):
        bundle = loader()
        X, y = bundle.data.astype(float), bundle.target.astype(int)
        for seed in range(seeds):
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.4, stratify=y, random_state=seed
            )
            X_cal, X_test, y_cal, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=10_000 + seed
            )
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_cal = scaler.transform(X_cal)
            X_test = scaler.transform(X_test)

            shifted_tests = {}
            for shift, levels in [("Gaussian noise", gaussian_levels), ("Feature masking", masking_levels)]:
                for level_index, severity in enumerate(levels):
                    corruption_seed = 1_000_000 * dataset_index + 10_000 * seed + 100 * level_index + (0 if shift == "Gaussian noise" else 1)
                    shifted_tests[(shift, severity)] = corrupt(X_test, shift, severity, corruption_seed)

            for model_name in MODEL_NAMES:
                model = make_model(model_name, seed, quick)
                model.fit(X_train, y_train)
                cal_probs = model.predict_proba(X_cal)
                temperature = fit_temperature(cal_probs, y_cal)

                for (shift, severity), X_shifted in shifted_tests.items():
                    raw_probs = model.predict_proba(X_shifted)
                    for calibration, probs in [
                        ("Raw", raw_probs),
                        ("Temperature scaled", apply_temperature(raw_probs, temperature)),
                    ]:
                        metrics, bins, risks = metric_row(probs, y_test)
                        base = {
                            "dataset": dataset_name,
                            "seed": seed,
                            "model": model_name,
                            "calibration": calibration,
                            "shift": shift,
                            "severity": severity,
                            "temperature": temperature,
                            "n_test": len(y_test),
                        }
                        metric_rows.append({**base, **metrics})
                        for bin_index, lower, upper, count, bin_conf, bin_acc in bins:
                            reliability_rows.append({
                                **base,
                                "bin": bin_index,
                                "bin_lower": lower,
                                "bin_upper": upper,
                                "count": count,
                                "bin_confidence": bin_conf,
                                "bin_accuracy": bin_acc,
                            })
                        for coverage, risk in risks:
                            risk_rows.append({**base, "coverage": coverage, "risk": risk})

            completed += 1
            print(f"Completed {completed}/{total}: {dataset_name}, split {seed + 1}/{seeds}", flush=True)

    return pd.DataFrame(metric_rows), pd.DataFrame(reliability_rows), pd.DataFrame(risk_rows)


def ci_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model", "calibration", "shift", "severity"]
    values = ["accuracy", "mean_confidence", "confidence_gap", "ece", "brier", "nll", "aurc"]
    means = metrics.groupby(keys, as_index=False)[values].mean()
    sems = metrics.groupby(keys, as_index=False)[values].sem().fillna(0)
    summary = means.copy()
    for value in values:
        summary[f"{value}_ci95"] = 1.96 * sems[value]
    return summary


def style_plots():
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig, name: str):
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_confidence_accuracy(metrics: pd.DataFrame):
    data = metrics[(metrics["calibration"] == "Raw") & (metrics["shift"] == "Gaussian noise")]
    grouped = data.groupby(["dataset", "model", "severity"], as_index=False)[["accuracy", "mean_confidence"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = grouped[grouped["dataset"] == dataset]
        for model in MODEL_NAMES:
            model_data = subset[subset["model"] == model]
            color = MODEL_COLORS[model]
            ax.plot(model_data["severity"], model_data["accuracy"], color=color, marker="o", label=f"{model}: accuracy")
            ax.plot(model_data["severity"], model_data["mean_confidence"], color=color, marker="s", linestyle="--", alpha=0.9, label=f"{model}: confidence")
        ax.set_title(dataset)
        ax.set_xlabel("Gaussian noise severity (SD)")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Mean value")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    fig.suptitle("Accuracy and confidence can deteriorate at different rates", y=0.98, fontweight="bold")
    fig.tight_layout(rect=(0, 0.22, 1, 0.93))
    save_figure(fig, "figure_1_confidence_accuracy")


def plot_calibration(metrics: pd.DataFrame):
    data = metrics[metrics["shift"] == "Gaussian noise"]
    grouped = data.groupby(["dataset", "model", "calibration", "severity"], as_index=False).ece.mean()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15), sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        subset = grouped[grouped["dataset"] == dataset]
        for model in MODEL_NAMES:
            for calibration, linestyle in [("Raw", "-"), ("Temperature scaled", "--")]:
                line = subset[(subset["model"] == model) & (subset["calibration"] == calibration)]
                ax.plot(line["severity"], line["ece"], color=MODEL_COLORS[model], linestyle=linestyle, marker="o" if calibration == "Raw" else "s", label=f"{model}: {calibration}")
        ax.set_title(dataset)
        ax.set_xlabel("Gaussian noise severity (SD)")
        ax.set_ylabel("Expected calibration error")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    fig.suptitle("Clean-data calibration under increasing feature corruption", y=0.98, fontweight="bold")
    fig.tight_layout(rect=(0, 0.22, 1, 0.93))
    save_figure(fig, "figure_2_calibration_under_shift")


def plot_heatmap(metrics: pd.DataFrame):
    maximum = metrics.groupby("shift")["severity"].transform("max")
    data = metrics[(metrics["calibration"] == "Raw") & (metrics["severity"] == maximum)]
    table = data.groupby(["dataset", "model", "shift"]).confidence_gap.mean().unstack("shift")
    table.index = [f"{dataset}\n{model}" for dataset, model in table.index]
    fig, ax = plt.subplots(figsize=(6.7, 5.0))
    sns.heatmap(table, annot=True, fmt="+.3f", center=0, cmap="vlag", linewidths=0.6, cbar_kws={"label": "Confidence - accuracy"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Overconfidence at maximum corruption", fontweight="bold", pad=10)
    save_figure(fig, "figure_3_overconfidence_heatmap")


def plot_risk_coverage(risk: pd.DataFrame):
    maximum = risk.groupby("shift")["severity"].transform("max")
    data = risk[(risk["calibration"] == "Raw") & (risk["shift"] == "Gaussian noise") & (risk["severity"] == maximum)]
    grouped = data.groupby(["dataset", "model", "coverage"], as_index=False).risk.mean()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.05), sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        subset = grouped[grouped["dataset"] == dataset]
        for model in MODEL_NAMES:
            line = subset[subset["model"] == model]
            ax.plot(line["coverage"], line["risk"], marker="o", color=MODEL_COLORS[model], label=model)
        ax.set_title(dataset)
        ax.set_xlabel("Coverage (fraction predicted)")
        ax.set_ylabel("Selective risk (error rate)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    fig.suptitle("Does abstaining on low-confidence cases reduce error under shift?", y=0.98, fontweight="bold")
    fig.tight_layout(rect=(0, 0.17, 1, 0.92))
    save_figure(fig, "figure_4_risk_coverage")


def plot_shift_comparison(metrics: pd.DataFrame):
    data = metrics[metrics["calibration"] == "Raw"].copy()
    data["relative_severity"] = data.groupby("shift")["severity"].transform(lambda x: x / x.max())
    grouped = data.groupby(["shift", "model", "relative_severity"], as_index=False)[["accuracy", "confidence_gap"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    for ax, metric, title in zip(axes, ["accuracy", "confidence_gap"], ["Accuracy", "Confidence - accuracy"]):
        for shift, linestyle in [("Gaussian noise", "-"), ("Feature masking", "--")]:
            for model in MODEL_NAMES:
                line = grouped[(grouped["shift"] == shift) & (grouped["model"] == model)]
                ax.plot(line["relative_severity"], line[metric], color=MODEL_COLORS[model], linestyle=linestyle, marker="o", label=f"{model}: {shift}")
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
        ax.set_xlabel("Normalized corruption severity")
        ax.set_ylabel(title)
        ax.set_title(title, fontweight="bold")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    fig.suptitle("Robustness across two controlled shift mechanisms", y=0.98, fontweight="bold")
    fig.tight_layout(rect=(0, 0.28, 1, 0.91))
    save_figure(fig, "figure_5_shift_comparison")


def generate_figures(metrics: pd.DataFrame, risk: pd.DataFrame):
    style_plots()
    plot_confidence_accuracy(metrics)
    plot_calibration(metrics)
    plot_heatmap(metrics)
    plot_risk_coverage(risk)
    plot_shift_comparison(metrics)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run two seeds with smaller ensembles for a fast smoke test.")
    parser.add_argument("--seeds", type=int, default=None, help="Override the number of repeated splits.")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else (2 if args.quick else 20)
    if seeds < 1:
        raise ValueError("--seeds must be at least 1")
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    start = time.time()
    print(f"Running {seeds} repeated splits across {len(DATASETS)} datasets...")
    metrics, reliability, risk = run_experiments(seeds, args.quick)
    summary = ci_summary(metrics)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    reliability.to_csv(RESULTS / "reliability.csv", index=False)
    risk.to_csv(RESULTS / "risk_coverage.csv", index=False)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    generate_figures(metrics, risk)
    elapsed = time.time() - start
    metadata = {
        "seeds": seeds,
        "quick": args.quick,
        "elapsed_seconds": elapsed,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "matplotlib": mpl.__version__,
            "seaborn": sns.__version__,
        },
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nDone in {elapsed:.1f} seconds.")
    print(f"Results: {RESULTS}")
    print(f"Figures: {FIGURES}")


if __name__ == "__main__":
    main()

"""Mechanism diagnostics for the controlled feature-corruption study.

Run after installing requirements:
    python diagnostics.py

This leaves the main experiment unchanged and writes one additional CSV and
one publication-ready figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from experiment import (
    DATASETS,
    FIGURES,
    MODEL_COLORS,
    MODEL_NAMES,
    RESULTS,
    corrupt,
    make_model,
    save_figure,
    style_plots,
)


SEVERITIES = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]


def decision_margin(model, X: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Return the model's top-class separation for each observation."""
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        if scores.ndim == 1:
            return np.abs(scores)
        ordered = np.sort(scores, axis=1)
        return ordered[:, -1] - ordered[:, -2]
    ordered = np.sort(probs, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def forest_vote_statistics(model, X: np.ndarray, prediction: np.ndarray):
    """Measure disagreement among individual random-forest trees."""
    tree_predictions = np.vstack([tree.predict(X).astype(int) for tree in model.estimators_])
    n_classes = len(model.classes_)
    counts = np.stack([(tree_predictions == label).sum(axis=0) for label in model.classes_], axis=1)
    frequencies = counts / len(model.estimators_)
    agreement = frequencies.max(axis=1)
    safe = np.clip(frequencies, 1e-12, 1.0)
    entropy = -(safe * np.log(safe)).sum(axis=1) / np.log(n_classes)
    return agreement, entropy


def safe_mean(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    selected = values if mask is None else values[mask]
    return float(np.mean(selected)) if len(selected) else float("nan")


def safe_correlation(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def run_diagnostics(seeds: int, quick: bool) -> pd.DataFrame:
    rows = []
    total = len(DATASETS) * seeds
    completed = 0

    for dataset_index, (dataset_name, loader) in enumerate(DATASETS.items()):
        bundle = loader()
        X, y = bundle.data.astype(float), bundle.target.astype(int)
        for seed in range(seeds):
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.4, stratify=y, random_state=seed
            )
            _, X_test, _, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=10_000 + seed
            )
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            shifted = {}
            for level_index, severity in enumerate(SEVERITIES):
                corruption_seed = 1_000_000 * dataset_index + 10_000 * seed + 100 * level_index
                shifted[severity] = corrupt(X_test, "Gaussian noise", severity, corruption_seed)

            for model_name in MODEL_NAMES:
                model = make_model(model_name, seed, quick)
                model.fit(X_train, y_train)
                for severity, X_shifted in shifted.items():
                    probs = model.predict_proba(X_shifted)
                    prediction = probs.argmax(axis=1)
                    confidence = probs.max(axis=1)
                    correct = prediction == y_test
                    error = ~correct
                    margin = decision_margin(model, X_shifted, probs)

                    row = {
                        "dataset": dataset_name,
                        "seed": seed,
                        "model": model_name,
                        "severity": severity,
                        "n_test": len(y_test),
                        "n_errors": int(error.sum()),
                        "accuracy": float(correct.mean()),
                        "confidence_all": safe_mean(confidence),
                        "confidence_correct": safe_mean(confidence, correct),
                        "confidence_error": safe_mean(confidence, error),
                        "margin_all": safe_mean(margin),
                        "margin_correct": safe_mean(margin, correct),
                        "margin_error": safe_mean(margin, error),
                        "high_confidence_error_rate": safe_mean(confidence >= 0.9, error),
                        "vote_agreement": float("nan"),
                        "vote_agreement_error": float("nan"),
                        "vote_entropy": float("nan"),
                        "vote_entropy_error": float("nan"),
                        "disagreement_confidence_correlation": float("nan"),
                    }

                    if model_name == "Random Forest":
                        agreement, entropy = forest_vote_statistics(model, X_shifted, prediction)
                        disagreement = 1.0 - agreement
                        row.update({
                            "vote_agreement": safe_mean(agreement),
                            "vote_agreement_error": safe_mean(agreement, error),
                            "vote_entropy": safe_mean(entropy),
                            "vote_entropy_error": safe_mean(entropy, error),
                            "disagreement_confidence_correlation": safe_correlation(disagreement, confidence),
                        })
                    rows.append(row)

            completed += 1
            print(f"Completed {completed}/{total}: {dataset_name}, split {seed + 1}/{seeds}", flush=True)

    return pd.DataFrame(rows)


def plot_diagnostics(data: pd.DataFrame):
    style_plots()
    grouped = data.groupby(["model", "severity"], as_index=False).agg(
        confidence_error=("confidence_error", "mean"),
        high_confidence_error_rate=("high_confidence_error_rate", "mean"),
        margin_error=("margin_error", "mean"),
        vote_entropy=("vote_entropy", "mean"),
        confidence_all=("confidence_all", "mean"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.25))

    for model in MODEL_NAMES:
        line = grouped[grouped["model"] == model]
        axes[0].plot(line["severity"], line["confidence_error"], marker="o", color=MODEL_COLORS[model], label=model)
        axes[1].plot(line["severity"], line["high_confidence_error_rate"], marker="o", color=MODEL_COLORS[model], label=model)

    axes[0].set_title("(a) Confidence on incorrect predictions")
    axes[0].set_ylabel("Mean confidence")
    axes[1].set_title("(b) High-confidence errors")
    axes[1].set_ylabel("Fraction of errors with confidence >= 0.9")

    forest = grouped[grouped["model"] == "Random Forest"]
    axes[2].plot(forest["severity"], forest["vote_entropy"], marker="o", color="#CC79A7", label="Tree-vote entropy")
    axes[2].plot(forest["severity"], forest["confidence_all"], marker="s", linestyle="--", color=MODEL_COLORS["Random Forest"], label="Forest confidence")
    axes[2].set_title("(c) Random-forest disagreement")
    axes[2].set_ylabel("Mean value")
    axes[2].legend(frameon=False, loc="best")

    for ax in axes:
        ax.set_xlabel("Gaussian noise severity (SD)")
        ax.set_ylim(0, 1.02)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.36, 0.01), ncol=3, frameon=False)
    fig.suptitle("Diagnostic evidence for model-specific confidence responses", y=0.98, fontweight="bold")
    fig.tight_layout(rect=(0, 0.16, 1, 0.92))
    save_figure(fig, "figure_6_mechanism_diagnostics")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run two splits with smaller ensembles.")
    parser.add_argument("--seeds", type=int, default=None, help="Override repeated splits.")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else (2 if args.quick else 20)
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    print(f"Running mechanism diagnostics with {seeds} repeated splits...")
    diagnostics = run_diagnostics(seeds, args.quick)
    diagnostics.to_csv(RESULTS / "diagnostics.csv", index=False)
    plot_diagnostics(diagnostics)
    print(f"\nDone. New result: {RESULTS / 'diagnostics.csv'}")
    print(f"New figure: {FIGURES / 'figure_6_mechanism_diagnostics.png'}")


if __name__ == "__main__":
    main()


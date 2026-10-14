"""Build tables and static scientific figures from saved experiment results."""
import json
import os
import sys
from pathlib import Path

import matplotlib
cache = Path(__file__).resolve().parents[2] / "results" / "temp" / "matplotlib"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.part1_pre_midterm.common import ROOT


def main():
    results = ROOT / "results"
    figures = results / "figures"
    tree = json.loads((results / "tree_validation_audit.json").read_text())
    mlp = json.loads((results / "mlp_validation_audit.json").read_text())
    ga = json.loads((results / "ga_validation_audit.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    labels = ["Adapted full", "Depth 6", "Leaf >=20", "sklearn full", "sklearn pruned"]
    x = np.arange(len(labels))
    axes[0].bar(x - .18, [r["train_macro_f1"] for r in tree["runs"]], .36, label="Train")
    axes[0].bar(x + .18, [r["validation_macro_f1"] for r in tree["runs"]], .36, label="Validation")
    axes[0].set(xticks=x, xticklabels=labels, ylim=(0, 1.05), ylabel="Six-class Macro-F1", title="Tree generalization")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].legend()
    axes[1].bar(labels, [r["leaves"] for r in tree["runs"]], color="#4786a6")
    axes[1].set(ylabel="Number of leaves", title="Tree complexity")
    axes[1].tick_params(axis="x", labelrotation=25)
    fig.savefig(figures / "tree_stopping_pruning.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for width, color in [(32, "#226ca5"), (64, "#c66d26")]:
        group = [r for r in mlp["runs"] if r["hidden_width"] == width]
        losses = np.array([[h["validation_loss"] for h in r["history"]] for r in group])
        train_losses = np.array([[h["train_loss"] for h in r["history"]] for r in group])
        epochs = np.arange(1, losses.shape[1] + 1)
        axes[0].plot(epochs, losses.mean(axis=0), color=color, label=f"Width {width}: validation")
        axes[0].fill_between(epochs, losses.min(axis=0), losses.max(axis=0), color=color, alpha=.14)
        axes[0].plot(epochs, train_losses.mean(axis=0), color=color, linestyle="--", label=f"Width {width}: train")
        axes[1].plot(np.arange(len(group)), [r["validation_macro_f1"] for r in group],
                     marker="o", linestyle="none", color=color, label=f"Width {width}")
    axes[0].set(xlabel="Epoch", ylabel="Mean cross-entropy", title="MLP losses: mean and seed range")
    axes[1].set(xlabel="Training seed", ylabel="Validation Macro-F1", title="Paired capacity comparison",
                xticks=[0, 1, 2], xticklabels=["42", "0", "1"])
    axes[0].legend(fontsize=8)
    axes[1].legend()
    fig.savefig(figures / "mlp_capacity.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for record in ga["runs"]:
        label = f"seed {record['seed']}, elite {record['elite_count']}"
        style = "-" if record["elite_count"] else "--"
        color = {42: "#226ca5", 0: "#c66d26", 1: "#398145"}[record["seed"]]
        axes[0].plot([h["generation"] for h in record["history"]],
                     [h["generation_best"] for h in record["history"]], style, color=color, label=label)
        axes[1].plot([h["generation"] for h in record["history"]],
                     [h["diversity"] for h in record["history"]], style, color=color, label=label)
    axes[0].set(xlabel="Generation (zero-based)", ylabel="Inner fitness", title="GA current-generation best")
    axes[1].set(xlabel="Generation (zero-based)", ylabel="Normalized mean Hamming distance", title="Population diversity", ylim=(0, .6))
    axes[0].legend(fontsize=7, ncol=2)
    fig.savefig(figures / "ga_elitism_diversity.png", dpi=170)
    plt.close(fig)
    lines = ["# Part I practice results - 2026-10-14", "",
             "All results below use the same outer train-subject validation split. No new final-test data was evaluated. These are dated Codex-assisted experiments; they do not replace historical HW2/HW3 scans or establish independent authorship.", "",
             "## Tree stopping and pruning", "",
             "| Configuration | Train F1 | Validation F1 | Validation accuracy | Depth | Leaves | Fit time (s) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in tree["runs"]:
        lines.append(f"| {r['model']} | {r['train_macro_f1']:.4f} | {r['validation_macro_f1']:.4f} | {r['validation_accuracy']:.4f} | {r['depth']} | {r['leaves']} | {r['train_time_s']:.2f} |")
    lines += ["", "Depth six reduces leaves from 100 to 38 and improves validation F1 from 0.8624 to 0.8719 in the adapted tree. Minimum leaf size 20 reduces complexity but lowers validation F1 to 0.8546. sklearn post-pruning reduces leaves from 100 to 65 but lowers validation F1 from 0.8815 to 0.8706. Regularization therefore does not uniformly improve this holdout. The implementations use different threshold/tie behavior and are not claimed to predict identically on HAR.", "",
              "## MLP capacity", "", "| Width | Seed | Train F1 | Validation F1 | Validation accuracy | Fit time (s) | Parameters |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in mlp["runs"]:
        lines.append(f"| {r['hidden_width']} | {r['seed']} | {r['train_macro_f1']:.4f} | {r['validation_macro_f1']:.4f} | {r['validation_accuracy']:.4f} | {r['train_time_s']:.2f} | {r['parameter_count']} |")
    lines += ["", "All runs use tanh, one hidden layer, existing NumPy forward/backward, mini-batch SGD, lr 0.05, 60 epochs and batch size 128. No early stopping or validation-selected epoch is used.", ""]
    for s in mlp["summary"]:
        lines.append(f"- Width {s['width']}: validation F1 mean {s['mean_validation_f1']:.6f}, sample standard deviation {s['std_validation_f1']:.6f}; mean fit time {s['mean_train_time_s']:.2f} s.")
    lines += ["", "The mean width advantage is about 0.001, smaller than the observed seed variability; 64 units are not consistently better (two seed pairs favor 32). More capacity increases train performance and runtime, without a demonstrated robust generalization gain. Seed variation is not population uncertainty. Hardware timings use one BLAS thread and are not directly comparable to the old HW3 Adam timings.", "",
              "## GA feature selection and elitism", "",
              "Candidates are the top 12 ANOVA features ranked on 3,911 windows from 11 inner-fit subjects. Fitness uses 1,359 windows from four separate inner-validation subjects. The chosen mask is refit on 5,270 outer-fit windows and evaluated on the untouched-by-search outer validation. All 12 candidates with GNB give outer validation F1 " + f"{ga['all_candidates_baseline']['validation_macro_f1']:.6f}.", "",
              "| Seed | Elites | Best inner fitness | Inner F1 | Features | Outer validation F1 | Unique fitness evaluations | Final diversity |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ga["runs"]:
        lines.append(f"| {r['seed']} | {r['elite_count']} | {r['best_inner_fitness']:.6f} | {r['best_inner_macro_f1']:.6f} | {r['selected_count']} | {r['validation_macro_f1']:.6f} | {r['fitness_evaluations']} | {r['history'][-1]['diversity']:.4f} |")
    lines += ["", "Population size 16; 12 evaluated generations; tournament size three; single-point crossover; bit mutation probability 1/12; fitness = inner F1 - 0.02 * count/12. Paired variants start with identical populations. Elites cannot mutate. Both variants return an all-time best, not necessarily the last generation's best.", "",
              "Elitism prevents a decline in current-generation best fitness but does not ensure a better discovered optimum. Seed 1 with one elite finds a lower inner-fitness 8-feature mask with higher outer F1 than the 7-feature mask found in other runs. This illustrates that inner fitness ranking does not perfectly transfer to outer subjects. GA gives feature compression rather than a demonstrated accuracy gain over the all-12 baseline. Three seeds and a small candidate pool do not prove global optimality.", "",
              "## BN/TAN checks and suitability", "",
              "The specified five-binary-variable worked network normalizes to one; enumeration agrees with all four d-separation examples and the 10-parameter count. This is an illustrative mathematical check, not a second predictive benchmark dataset.", "",
              "In HAR fit rows, `tBodyAccMag-mean()` and `tBodyAccMag-energy()` have within-class Pearson correlations 0.868--0.988. This is empirical evidence against strict conditional independence for this pair, not proof that a particular fitted TAN will improve classification. TAN could capture one feature dependency per node, but a continuous TAN requires conditional density estimation; discretization would require fit-only binning and add information loss. No fitted TAN performance is claimed.", "",
              "## Artifacts and reproduction", "",
              "Settings and hypotheses were recorded before running in `docs/experiments/PART1_PLAN.md`. Reference attribution, exact saved source hashes and equation-to-function mapping are in `docs/reference/DISSECTION.md` and `sources.json`. Full per-run confusion matrices, feature names, losses, diversity and runtime are in the audit JSON files. Curves are in `results/figures/`.", "",
              "```text", "python experiments/part1_pre_midterm/w05_tree.py", "python experiments/part1_pre_midterm/w05_mlp.py",
              "python experiments/part1_pre_midterm/w07_ga.py", "python experiments/part1_pre_midterm/w07_bn.py",
              "python experiments/part1_pre_midterm/summarize_practice.py", "```", "",
              "No independent pre-reference W07 code attempt or closed-book reproduction is claimed. Validation reuse, the historical test exposure documented in `PART1_METRIC_AUDIT.md`, and the small seed count limit conclusions. The two final PDF documents remain deferred."]
    (results / "PART1_PRACTICE_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved practice summary and three figures.")


if __name__ == "__main__":
    main()

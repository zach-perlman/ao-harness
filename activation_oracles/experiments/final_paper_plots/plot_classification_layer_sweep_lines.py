import json
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Text sizes for plots (edit here to change all text sizes)
# ---------------------------------------------------------------------------
FONT_SIZE_SUBPLOT_TITLE = 20  # Subplot titles (e.g., "Taboo", "Gender", "Secret Keeping")
FONT_SIZE_Y_AXIS_LABEL = 18  # Y-axis labels (e.g., "Average Accuracy")
FONT_SIZE_Y_AXIS_TICK = 16  # Y-axis tick labels (numbers on y-axis)
FONT_SIZE_BAR_VALUE = 16  # Numbers above each bar
FONT_SIZE_LEGEND = 18  # Legend text size

# Datasets to average
IID_DATASETS = [
    "geometry_of_truth",
    "relations",
    "sst2",
    "md_gender",
    "snli",
    "ner",
    "tense",
]

OOD_DATASETS = [
    "ag_news",
    "language_identification",
    "singular_plural",
    "engels_headline_istrump",
    "engels_headline_isobama",
    "engels_headline_ischina",
    "engels_hist_fig_ismale",
]

# Which LoRA result to track for each base model (substring must match exactly one JSON per layer dir)
MODEL_CONFIG = {
    # Activation oracle / full-dataset naming differs by model, so allow multiple substrings per model.
    "Llama-3.3-70B-Instruct": {
        "prefix": "classification_Llama-3_3-70B-Instruct_single_token_",
        "keywords": ["act_cls_latentqa_pretrain_mix"],
    },
    "Qwen3-8B": {
        "prefix": "classification_Qwen3-8B_single_token_",
        "keywords": ["latentqa_cls_past_lens", "act_cls_latentqa_pretrain_mix"],
    },
    "Gemma-2-9B-IT": {
        "prefix": "classification_gemma-2-9b-it_single_token_",
        "keywords": ["latentqa_cls_past_lens", "act_cls_latentqa_pretrain_mix"],
    },
}

ROOT = Path("experiments/classification_layer_sweep")
OUT_DIR = Path("images/layer_comparison")
TRAINED_PERCENTS = {25, 50, 75}


def accuracy(records, dataset_ids):
    selected = [r for r in records if r["dataset_id"] in dataset_ids]
    assert selected, "No records for requested split"
    correct = sum(1 for r in selected if r["target"].lower().strip() in r["ground_truth"].lower().strip())
    return correct / len(selected)


def load_split_acc(folder: Path, keywords):
    chosen = None
    for keyword in keywords:
        matches = [p for p in folder.glob("*.json") if keyword in p.name]
        if len(matches) == 1:
            chosen = matches[0]
            break
        if len(matches) > 1:
            raise AssertionError(f"Keyword '{keyword}' matched multiple files in {folder}: {matches}")
    assert chosen is not None, f"No file matched any of {keywords} in {folder}"

    data = json.loads(chosen.read_text())
    records = data["records"]
    return accuracy(records, IID_DATASETS), accuracy(records, OOD_DATASETS), data["meta"]["layer_percent"]


def load_model_data(model_name: str, prefix: str, keywords):
    layer_dirs = sorted(ROOT.glob(f"{prefix}*"))
    assert layer_dirs, f"No layer sweep dirs found for prefix {prefix}"

    layers = []
    iid_vals = []
    ood_vals = []

    for layer_dir in layer_dirs:
        iid_acc, ood_acc, percent = load_split_acc(layer_dir, keywords)
        layers.append(percent)
        iid_vals.append(iid_acc)
        ood_vals.append(ood_acc)

    # Sort by percent in case glob order differs
    order = sorted(range(len(layers)), key=lambda i: layers[i])
    layers = [layers[i] for i in order]
    iid_vals = [iid_vals[i] for i in order]
    ood_vals = [ood_vals[i] for i in order]

    return layers, iid_vals, ood_vals


def split_points(x_list, y_list):
    trained_x, trained_y, untrained_x, untrained_y = [], [], [], []
    for x, y in zip(x_list, y_list):
        if x in TRAINED_PERCENTS:
            trained_x.append(x)
            trained_y.append(y)
        else:
            untrained_x.append(x)
            untrained_y.append(y)
    return trained_x, trained_y, untrained_x, untrained_y


def plot_all_models():
    # Colors for each model - using distinct colors
    model_colors = {
        "Llama-3.3-70B-Instruct": "#1f77b4",  # blue
        "Qwen3-8B": "#2ca02c",  # green
        "Gemma-2-9B-IT": "#d62728",  # red
    }

    plt.figure(figsize=(10, 6))

    for model_name, cfg in MODEL_CONFIG.items():
        layers, iid_vals, ood_vals = load_model_data(model_name, cfg["prefix"], cfg["keywords"])
        color = model_colors[model_name]

        # Plot lines
        iid_line = plt.plot(
            layers, iid_vals, color=color, linewidth=2, linestyle="-", alpha=0.7, label=f"{model_name} IID"
        )[0]
        ood_line = plt.plot(
            layers, ood_vals, color=color, linewidth=2, linestyle="--", alpha=0.7, label=f"{model_name} OOD"
        )[0]

        # Split points into trained and untrained
        iid_tx, iid_ty, iid_ux, iid_uy = split_points(layers, iid_vals)
        ood_tx, ood_ty, ood_ux, ood_uy = split_points(layers, ood_vals)

        # IID markers: filled for trained, hollow for untrained
        plt.scatter(iid_tx, iid_ty, color=color, marker="o", s=50, zorder=5, edgecolors=color, linewidths=1.5)
        plt.scatter(iid_ux, iid_uy, facecolors="white", edgecolors=color, marker="o", s=50, zorder=6, linewidths=1.5)

        # OOD markers: filled for trained, hollow for untrained
        plt.scatter(ood_tx, ood_ty, color=color, marker="s", s=50, zorder=5, edgecolors=color, linewidths=1.5)
        plt.scatter(ood_ux, ood_uy, facecolors="white", edgecolors=color, marker="s", s=50, zorder=6, linewidths=1.5)

    # Add legend entries for trained/untrained distinction
    hollow_handle = plt.Line2D(
        [],
        [],
        color="black",
        marker="o",
        linestyle="None",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=8,
        label="Not trained",
    )
    solid_handle = plt.Line2D(
        [], [], color="black", marker="o", linestyle="None", markerfacecolor="black", markersize=8, label="Trained"
    )

    plt.xlabel("Layer Percent", fontsize=FONT_SIZE_Y_AXIS_LABEL)
    plt.ylabel("Classification Accuracy", fontsize=FONT_SIZE_Y_AXIS_LABEL)
    plt.ylim(0, 1.05)
    plt.tick_params(axis="x", labelsize=FONT_SIZE_Y_AXIS_TICK)
    plt.tick_params(axis="y", labelsize=FONT_SIZE_Y_AXIS_TICK)
    plt.grid(alpha=0.3)
    # plt.title("Classification Layer Sweep - All Models", fontsize=FONT_SIZE_SUBPLOT_TITLE)

    # Get all existing legend handles and add trained/untrained distinction
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([solid_handle, hollow_handle])
    plt.legend(handles=handles, loc="best", fontsize=FONT_SIZE_LEGEND)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "all_models_layer_sweep_lines"
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved {out_path.with_suffix('.pdf')}")
    print(f"Saved {out_path.with_suffix('.png')}")
    plt.close()


def main():
    plot_all_models()


if __name__ == "__main__":
    main()

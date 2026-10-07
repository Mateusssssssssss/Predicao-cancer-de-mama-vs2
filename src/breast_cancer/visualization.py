from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def save_eda_figures(frame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = frame.drop(columns=["id", "Unnamed: 32"], errors="ignore").copy()
    features = clean.drop(columns="diagnosis")

    clean.select_dtypes("number").hist(figsize=(15, 12), bins=30)
    plt.suptitle("Distribuição das variáveis")
    plt.tight_layout()
    plt.savefig(output_dir / "distribuicao_dados.png", dpi=150)
    plt.close("all")

    plt.figure(figsize=(14, 8))
    sns.boxplot(data=features, orient="h", color="skyblue")
    plt.title("Boxplot das variáveis numéricas")
    plt.tight_layout()
    plt.savefig(output_dir / "boxplot.png", dpi=150)
    plt.close()

    numeric = clean.copy()
    numeric["diagnosis"] = numeric["diagnosis"].map({"B": 0, "M": 1})
    correlations = numeric.corr(numeric_only=True)["diagnosis"].drop("diagnosis").sort_values()
    ax = correlations.plot(kind="barh", figsize=(10, 8))
    ax.set(title="Correlação das Variáveis com o Diagnosis", xlabel="Correlação", ylabel="Variáveis")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "correlacao_variaveis.png", dpi=150)
    plt.close()

    clean["diagnosis"].value_counts().reindex(["B", "M"]).plot(
        kind="bar", color=["skyblue", "lightcoral"], figsize=(7, 5)
    )
    plt.title("Distribuição das Classes de Diagnóstico")
    plt.xlabel("Diagnóstico (B=Benigno, M=Maligno)")
    plt.ylabel("Quantidade")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "distribuicao_classes.png", dpi=150)
    plt.close()


def save_roc_curve(fpr, tpr, auc_value: float, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_value:.3f})", color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], "--", color="navy")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.grid(linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

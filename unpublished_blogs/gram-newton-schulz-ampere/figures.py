from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def save_figure(figure: Figure, output_stem: Path) -> None:
    figure.tight_layout()
    for extension in ("svg", "pdf", "png"):
        figure.savefig(output_stem.with_suffix(f".{extension}"))
    plt.close(figure)


def plot_modal_speedup(data_directory: Path, figure_directory: Path) -> None:
    timings = pd.read_csv(data_directory / "end_to_end.csv")
    modal_timings = timings[timings["evidence"] == "Modal final implementation"]
    figure, axis = plt.subplots()
    sns.lineplot(
        data=modal_timings,
        x="batch_size",
        y="standard_over_triangular_speedup",
        hue="device",
        marker="o",
        ax=axis,
    )
    axis.set(
        xlabel="Batch size",
        ylabel="Speedup over upstream eager Standard NS",
        title="Single-GPU actual-shape speedup on Ampere",
    )
    save_figure(figure, figure_directory / "modal_speedup")


def plot_a100_latency(data_directory: Path, figure_directory: Path) -> None:
    timings = pd.read_csv(data_directory / "end_to_end.csv")
    a100_timings = timings[timings["evidence"] == "Turing cleanup regression"].rename(
        columns={
            "upstream_standard_eager_ms": "Upstream Standard NS",
            "torch_gns_eager_ms": "Torch GNS",
            "triangular_cutlass_gns_ms": "Triangular CUTLASS GNS",
        }
    )
    long_timings = a100_timings.melt(
        id_vars="batch_size",
        value_vars=(
            "Upstream Standard NS",
            "Torch GNS",
            "Triangular CUTLASS GNS",
        ),
        var_name="Method",
        value_name="Median latency (ms)",
    )
    figure, axis = plt.subplots()
    sns.lineplot(
        data=long_timings,
        x="batch_size",
        y="Median latency (ms)",
        hue="Method",
        marker="o",
        ax=axis,
    )
    axis.set(
        xlabel="Batch size",
        ylabel="Median latency (ms)",
        title="One-A100 latency at B × 16384 × 2048",
    )
    save_figure(figure, figure_directory / "a100_latency")


def plot_primitive_speedup(data_directory: Path, figure_directory: Path) -> None:
    timings = pd.read_csv(data_directory / "primitives_turing_a100.csv")
    operation_names = {
        "symmetric_square_baddbmm": "Symmetric square",
        "symmetric_gram": "Gram product",
        "full_gns_core": "GNS core",
    }
    rows: list[dict[str, float | int | str]] = []
    for (operation, batch_size), group in timings.groupby(["kind", "batch_size"]):
        torch_time = float(group[group["name"] == "torch"]["median_ms"].iloc[0])
        cutlass_time = float(
            group[group["name"] == "cutlass_triangular"]["median_ms"].iloc[0]
        )
        rows.append(
            {
                "Operation": operation_names[str(operation)],
                "Batch size": int(batch_size),
                "Speedup": torch_time / cutlass_time,
            }
        )
    speedups = pd.DataFrame(rows)
    figure, axis = plt.subplots()
    sns.barplot(
        data=speedups,
        x="Operation",
        y="Speedup",
        hue="Batch size",
        ax=axis,
    )
    axis.set(
        xlabel="Operation",
        ylabel="Speedup over Torch",
        title="Contribution of the triangular symmetric kernel",
    )
    save_figure(figure, figure_directory / "primitive_speedup")


def plot_backend_profile(data_directory: Path, figure_directory: Path) -> None:
    profile = pd.read_csv(data_directory / "generic_backend_profile.csv")
    batch_thirty_two = profile[profile["batch_size"] == 32].copy()
    batch_thirty_two["Operation"] = batch_thirty_two["kind"].map(
        {
            "square_baddbmm": "Square",
            "gram_bmm": "Gram",
            "tall_bmm": "Rectangular",
        }
    )
    figure, axis = plt.subplots()
    sns.barplot(
        data=batch_thirty_two,
        x="Operation",
        y="median_ms",
        hue="backend",
        ax=axis,
    )
    axis.set_yscale("log")
    axis.set(
        xlabel="Operation",
        ylabel="Median latency (ms, logarithmic scale)",
        title="Generic backend profile on one A100 at B = 32",
    )
    save_figure(figure, figure_directory / "generic_backend_profile")


def plot_algorithmic_work(data_directory: Path, figure_directory: Path) -> None:
    work = pd.read_csv(data_directory / "algorithmic_work.csv")
    figure, axis = plt.subplots()
    sns.barplot(
        data=work,
        x="algorithm",
        y="teraflops_per_matrix",
        ax=axis,
    )
    axis.set(
        xlabel="Algorithm",
        ylabel="TFLOP per matrix",
        title="Algorithmic work at 16384 × 2048",
    )
    axis.tick_params(axis="x", rotation=15)
    save_figure(figure, figure_directory / "algorithmic_work")


def main() -> None:
    article_directory = Path(__file__).resolve().parent
    data_directory = article_directory / "data"
    figure_directory = article_directory / "figures"
    figure_directory.mkdir(exist_ok=True)
    plot_modal_speedup(data_directory, figure_directory)
    plot_a100_latency(data_directory, figure_directory)
    plot_primitive_speedup(data_directory, figure_directory)
    plot_backend_profile(data_directory, figure_directory)
    plot_algorithmic_work(data_directory, figure_directory)


if __name__ == "__main__":
    main()

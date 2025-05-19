import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np

def parse_json_logs(json_logs, metric):
    """Parse JSON logs to extract metric values for each epoch."""
    log_data = {}
    for log_file in json_logs:
        with open(log_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                if 'epoch' not in entry or metric not in entry:
                    continue
                epoch = entry['epoch']
                if epoch not in log_data:
                    log_data[epoch] = []
                log_data[epoch].append(entry[metric])
    return log_data

def calculate_epoch_average(log_data):
    """Calculate the average value of the metric for each epoch."""
    epoch_avg = {}
    for epoch, values in log_data.items():
        epoch_avg[epoch] = np.mean(values)
    return epoch_avg

def plot_epoch_curve(epoch_avg, title, xlabel, ylabel, legend, out_file):
    """Plot the curve based on epoch data."""
    sns.set(style="darkgrid")
    epochs = sorted(epoch_avg.keys())
    metrics = [epoch_avg[epoch] for epoch in epochs]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics, marker='o', label=legend)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    if out_file:
        plt.savefig(out_file)
        print(f"Plot saved to {out_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot metric curve based on epoch.")
    parser.add_argument(
        "json_logs",
        type=str,
        nargs='+',
        help="Path(s) to JSON log file(s).")
    parser.add_argument(
        "--metric",
        type=str,
        default="mAP",
        help="Metric to plot, e.g., 'mAP'.")
    parser.add_argument(
        "--title",
        type=str,
        default="Metric Curve",
        help="Title of the plot.")
    parser.add_argument(
        "--xlabel",
        type=str,
        default="Epoch",
        help="Label for the x-axis.")
    parser.add_argument(
        "--ylabel",
        type=str,
        default="Metric Value",
        help="Label for the y-axis.")
    parser.add_argument(
        "--legend",
        type=str,
        default="Metric",
        help="Legend for the plot.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to save the plot (e.g., 'output.png').")

    args = parser.parse_args()

    # Parse logs and calculate averages
    log_data = parse_json_logs(args.json_logs, args.metric)
    epoch_avg = calculate_epoch_average(log_data)

    # Plot the curve
    plot_epoch_curve(
        epoch_avg,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        legend=args.legend,
        out_file=args.out
    )

if __name__ == "__main__":
    main()

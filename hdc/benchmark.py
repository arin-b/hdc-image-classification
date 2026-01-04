import time
import json
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from torch.utils.data import Subset, DataLoader
from memory_profiler import memory_usage
from tqdm import tqdm

from features import FeatureExtractor, get_image_transform
from projection import generate_random_projection, encode_hypervector
from classifier import HDCClassifier

# =========================
# CONFIGURATION
# =========================
CLASSES = [0, 8]               # airplane vs ship
HD_DIM = 4096
CNN_OUTPUT_DIM = 128
SEED = 42

TRAIN_SIZES = [4, 8, 16, 32, 64, 128, 256]
TEST_SAMPLES = 200
LATENCY_RUNS = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# DATA LOADING
# =========================
def get_loaders(train_per_class):
    transform = get_image_transform()

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    def build_loader(dataset, limit):
        indices, counts = [], {k: 0 for k in CLASSES}
        for i, (_, label) in enumerate(dataset):
            if label in CLASSES and counts[label] < limit:
                indices.append(i)
                counts[label] += 1
        return DataLoader(Subset(dataset, indices), batch_size=1, shuffle=True)

    return (
        build_loader(train_set, train_per_class),
        build_loader(test_set, TEST_SAMPLES),
    )

# =========================
# TRAINING
# =========================
def train_hdc(train_loader):
    model = FeatureExtractor(CNN_OUTPUT_DIM).to(DEVICE).eval()
    R = generate_random_projection(CNN_OUTPUT_DIM, HD_DIM, SEED)

    clf = HDCClassifier(num_classes=len(CLASSES), hd_dim=HD_DIM)
    label_map = {orig: i for i, orig in enumerate(CLASSES)}

    with torch.no_grad():
        for img, label in train_loader:
            img = img.to(DEVICE)
            feat = model(img).cpu().numpy().flatten()
            h = encode_hypervector(feat, R)
            clf.add_sample(label_map[label.item()], h)

    return clf, model, R

# =========================
# ACCURACY
# =========================
def evaluate_accuracy(clf, model, R, test_loader):
    correct, total = 0, 0
    label_map = {orig: i for i, orig in enumerate(CLASSES)}

    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(DEVICE)
            feat = model(img).cpu().numpy().flatten()
            h = encode_hypervector(feat, R)
            pred = clf.predict(h)
            correct += int(pred == label_map[label.item()])
            total += 1

    return correct / total

# =========================
# LATENCY
# =========================
def benchmark_latency(model, clf, R, sample_img):
    feat_times, enc_times, inf_times = [], [], []

    # warm-up
    for _ in range(10):
        model(sample_img)

    for _ in range(LATENCY_RUNS):
        t0 = time.perf_counter()
        feat = model(sample_img)
        feat_times.append(time.perf_counter() - t0)

        feat_np = feat.cpu().numpy().flatten()
        t0 = time.perf_counter()
        h = encode_hypervector(feat_np, R)
        enc_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        clf.predict(h)
        inf_times.append(time.perf_counter() - t0)

    return {
        "feature_ms": np.median(feat_times) * 1e3,
        "encode_ms": np.median(enc_times) * 1e3,
        "infer_ms": np.median(inf_times) * 1e3,
    }

# =========================
# RAM MEASUREMENT
# =========================
def measure_ram(train_loader):
    def wrapped():
        train_hdc(train_loader)
    return memory_usage(wrapped, max_usage=True)

# =========================
# MAIN BENCHMARK LOOP
# =========================
def main():
    results = []

    for N in TRAIN_SIZES:
        print(f"\n=== Benchmarking: {N} samples / class ===")
        train_loader, test_loader = get_loaders(N)

        ram_mb = measure_ram(train_loader)

        clf, model, R = train_hdc(train_loader)
        acc = evaluate_accuracy(clf, model, R, test_loader)

        sample_img, _ = next(iter(test_loader))
        sample_img = sample_img.to(DEVICE)

        latency = benchmark_latency(model, clf, R, sample_img)

        results.append({
            "samples_per_class": N,
            "accuracy": acc,
            "ram_mb": ram_mb,
            **latency,
        })

        print(f"Accuracy: {acc:.3f}, RAM: {ram_mb:.1f} MB")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_results(results)

# =========================
# PLOTTING
# =========================
def plot_results(results):
    N = [r["samples_per_class"] for r in results]
    acc = [r["accuracy"] for r in results]
    ram = [r["ram_mb"] for r in results]

    feat = [r["feature_ms"] for r in results]
    enc = [r["encode_ms"] for r in results]
    inf = [r["infer_ms"] for r in results]

    # Accuracy plot
    plt.figure()
    plt.plot(N, acc, marker="o")
    plt.xlabel("Samples per class")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Training Samples")
    plt.grid()
    plt.savefig("accuracy_vs_samples.png")

    # RAM plot
    plt.figure()
    plt.plot(N, ram, marker="o", color="orange")
    plt.xlabel("Samples per class")
    plt.ylabel("Peak RAM (MB)")
    plt.title("RAM Usage vs Training Samples")
    plt.grid()
    plt.savefig("ram_vs_samples.png")

    # Latency plot
    plt.figure()
    plt.plot(N, feat, label="Feature")
    plt.plot(N, enc, label="Encode")
    plt.plot(N, inf, label="Inference")
    plt.xlabel("Samples per class")
    plt.ylabel("Latency (ms)")
    plt.title("Latency Breakdown")
    plt.legend()
    plt.grid()
    plt.savefig("latency_breakdown.png")

    print("\nPlots saved:")
    print("- accuracy_vs_samples.png")
    print("- ram_vs_samples.png")
    print("- latency_breakdown.png")

# =========================
if __name__ == "__main__":
    main()


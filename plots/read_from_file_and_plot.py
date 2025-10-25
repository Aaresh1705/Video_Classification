import re
import matplotlib.pyplot as plt

def extract_accuracies(filename):
    epochs = []
    train_acc = []
    test_acc = []
    
    # Regex to extract epoch number and accuracies
    pattern = re.compile(
        r"^\s*(\d+).*?Accuracy train:\s*([\d.]+)%\s*test:\s*([\d.]+)%"
    )

    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                train = float(match.group(2))
                test = float(match.group(3))
                epochs.append(epoch)
                train_acc.append(train)
                test_acc.append(test)

    return epochs, train_acc, test_acc


def plot_accuracies(epochs, train_acc, test_acc, output_file="accuracy_plot.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, 'o-', label='Train Accuracy', color='blue')
    plt.plot(epochs, test_acc, 's--', label='Test Accuracy', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Test Accuracy")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()  # Close figure to free memory
    print(f"✅ Plot saved as '{output_file}'")


if __name__ == "__main__":
    log_file = "50_epochs_simple_model.txt"  # Change this to your log file
    output_file = "accuracy_plot.png"  # Change if you want a different name

    epochs, train_acc, test_acc = extract_accuracies(log_file)

    if not epochs:
        print("⚠️ No accuracy data found in file.")
    else:
        print(f"Extracted {len(epochs)} epochs.")
        plot_accuracies(epochs, train_acc, test_acc, output_file)


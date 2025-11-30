"""
plot_data.py
-------------
Analyzes and visualizes the EEG class distribution
after preprocessing.
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_class_distribution(labels):
    """
    Analyze and plot class distribution from label data.
    """

    # Count samples per class
    unique_classes, class_counts = np.unique(labels, return_counts=True)

    # Optional mapping for better readability
    class_names = {
        0: "Focused",
        1: "Unfocused",
        2: "Drowsy"
    }
    label_names = [class_names.get(cls, str(cls)) for cls in unique_classes]

    # Print the class sample counts
    print(" Class Distribution:")
    for cls, count in zip(label_names, class_counts):
        print(f"  {cls:<10}: {count} samples")

    # ----------------------------------------------------
    # Bar Chart
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.bar(label_names, class_counts, color=['#4CAF50', '#FFC107', '#F44336'])
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.title("EEG Class Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

    # ----------------------------------------------------
    # Pie Chart
    # ----------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.pie(
        class_counts,
        labels=label_names,
        autopct='%1.1f%%',
        startangle=140,
        colors=['#4CAF50', '#FFC107', '#F44336']
    )
    plt.title("Class Distribution (Pie Chart)")
    plt.axis('equal')  # Equal aspect ratio ensures circular pie
    plt.show()


if __name__ == "__main__":
    # Example usage:
    # Import labels from preprocessing if saved as a .npy file or returned in memory
    from preprocessing import preprocess_data
    
    data = preprocess_data()
    labels = data["labels"]

    analyze_class_distribution(labels)

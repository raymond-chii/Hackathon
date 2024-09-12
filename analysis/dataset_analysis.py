import matplotlib.pyplot as plt
import pandas as pd


def analyze_emotion_dataset(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Select emotion columns
    emotion_columns = [
        col
        for col in df.columns
        if col.startswith("Answer.f1.") and col.endswith(".raw")
    ]

    # Function to get the emotion name from column name
    def get_name(col):
        return col.split(".")[2].capitalize()

    # Calculate emotion frequencies
    emotion_frequencies = df[emotion_columns].sum().sort_values(ascending=True)
    emotion_names = [get_name(col) for col in emotion_frequencies.index]

    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(emotion_names, emotion_frequencies)

    # Customize the plot
    plt.title("Frequency of Emotions in Dataset", fontsize=16)
    plt.xlabel("Number of Occurrences", fontsize=12)
    plt.ylabel("Emotions", fontsize=12)

    # Add value labels to the end of each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" {width}",
            ha="left",
            va="center",
        )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("emotion_frequency.png", dpi=300, bbox_inches="tight")
    print("Analysis complete. Visualization saved as 'emotion_frequency.png'.")


# Usage
analyze_emotion_dataset("../data.csv")

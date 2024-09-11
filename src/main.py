from mood_classifier import EmotionClassifier


def main():
    # Initialize and train the classifier
    classifier = EmotionClassifier("data.csv")
    X_test, y_test = classifier.train_model()

    # Evaluate the model
    evaluation_report = classifier.evaluate_model(X_test, y_test)
    print("Classification Report:\n", evaluation_report)

    # Example usage
    example_entry = "Today was a mix of emotions. I felt excited about my new project, but also a bit anxious about the upcoming deadline. Overall, I'm satisfied with my progress."
    print("\nPredicted emotions for the example entry:")
    display_emotions(classifier.predict_emotions(example_entry))

    # Interactive input
    while True:
        user_entry = input("\nEnter your journal entry (or 'quit' to exit): ")
        if user_entry.lower() == "quit":
            break

        results = classifier.predict_emotions(user_entry)
        print("\nPredicted emotions for your entry:")
        display_emotions(results)


def display_emotions(emotions):
    for emotion, percentage in emotions:
        print(f"{emotion}: {percentage:.2f}%")


if __name__ == "__main__":
    main()

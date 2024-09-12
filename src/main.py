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
    process_entry(classifier, example_entry)

    # Interactive input
    while True:
        user_entry = input("\nEnter your journal entry (or 'quit' to exit): ")
        if user_entry.lower() == "quit":
            break

        process_entry(classifier, user_entry)


def process_entry(classifier, entry):
    emotions = classifier.predict_emotions(entry)
    print("\nPredicted emotions for your entry:")
    display_emotions(emotions)

    color = classifier.mix_colors(emotions)
    print(f"\nMixed color representing emotional state: {color}")

    print("\nSuggested activities based on your emotions:")
    activities = classifier.suggest_activities(emotions)
    display_activities(activities)


def display_emotions(emotions):
    for emotion, percentage in emotions:
        print(f"{emotion}: {percentage:.2f}%")


def display_activities(activities):
    for i, activity in enumerate(activities, 1):
        print(f"{i}. {activity}")


if __name__ == "__main__":
    main()

from mood_classifier import EmotionClassifier

def main():
    # Initialize and train the classifier
    classifier = EmotionClassifier('data.csv')
    X_test, y_test = classifier.train_model()

    # Evaluate the model
    evaluation_report = classifier.evaluate_model(X_test, y_test)
    print("Classification Report:\n", evaluation_report)

    while True:
        print("\n1. Enter a new journal entry")
        print("2. View past entries")
        print("3. Quit")
        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            user_entry = input("\nEnter your journal entry: ")
            try:
                process_entry(classifier, user_entry)
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Please try again or contact support if the issue persists.")
        elif choice == '2':
            view_past_entries(classifier)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

    classifier.close_db()

def process_entry(classifier, entry):
    emotions, color, activities = classifier.process_entry(entry)
    if emotions is None:
        print("\nNo significant emotions detected or empty entry. Please try again with a more detailed entry.")
        return

    print("\nPredicted emotions for your entry:")
    display_emotions(emotions)
    
    print(f"\nMixed color representing emotional state: {color}")
    
    print("\nSuggested activities based on your emotions:")
    display_activities(activities)

def view_past_entries(classifier):
    entries = classifier.get_past_entries()
    if not entries:
        print("\nNo past entries found.")
        return

    for entry in entries:
        print(f"\nDate: {entry[1]}")
        print(f"Entry: {entry[2]}")
        print(f"Color: {entry[3]}")
        print("Emotions:")
        for emotion_data in entry[4].split(','):
            emotion, percentage = emotion_data.split(':')
            print(f"  {emotion}: {float(percentage):.2f}%")
        print("Suggested Activities:")
        for activity in entry[5].split(','):
            print(f"  - {activity}")
        print("-" * 50)

def display_emotions(emotions):
    for emotion, percentage in emotions:
        print(f"{emotion}: {percentage:.2f}%")

def display_activities(activities):
    for i, activity in enumerate(activities, 1):
        print(f"{i}. {activity}")

if __name__ == "__main__":
    main()
from train_model import train_eeg_model
from game import play_game

if __name__ == "__main__":
    print(" Training EEG model...")
    model, X_test, y_test = train_eeg_model()
    print("\n Launching car game based on EEG model predictions...")
    play_game(X_test, model)

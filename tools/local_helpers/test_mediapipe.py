import mediapipe as mp
try:
    print(f"Mediapipe version: {mp.__version__}")
    print(f"Solutions: {mp.solutions}")
    print("Mediapipe solutions accessed successfully.")
except AttributeError as e:
    print(f"Error accessing solutions: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

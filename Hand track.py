import cv2
import mediapipe as mp
import os
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def capture_images():
    gesture_name = input("Enter Gesture Name: ")
    save_path = f'dataset/{gesture_name}'
    os.makedirs(save_path, exist_ok=True)

    img_count = 0
    max_images = 100

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                h, w, c = img.shape
                landmarks = []
                for lm in handLms.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append(cx)
                    landmarks.append(cy)
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

                if img_count < max_images:
                    img_count += 1
                    print(f"Saving Image {img_count}/{max_images}")
                    np.save(f"{save_path}/{img_count}", np.array(landmarks))

        cv2.imshow("Capturing Images", img)
        if cv2.waitKey(1) & 0xFF == ord('q') or img_count >= max_images:
            break

def train_model():
    data = []
    labels = []

    gestures = os.listdir('dataset')
    for gesture in gestures:
        files = os.listdir(f'dataset/{gesture}')
        for file in files:
            arr = np.load(f'dataset/{gesture}/{file}')
            data.append(arr)
            labels.append(gesture)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(data, labels)

    with open('model.pkl', 'wb') as f:
        pickle.dump(knn, f)

    print("Model Trained and Saved!")


def detect():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                h, w, c = img.shape
                landmarks = []
                for lm in handLms.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append(cx)
                    landmarks.append(cy)

                if len(landmarks) == 42:  # 21 points x 2 (x, y)
                    prediction = model.predict([landmarks])
                    print("Detected Gesture:", prediction[0])
                    cv2.putText(img, prediction[0], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow("Detection Mode", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("Choose Mode:")
print("1. Create Database")
print("2. Train Model")
print("3. Start Detection")

choice = input("Enter Choice (1/2/3): ")

if choice == '1':
    capture_images()
elif choice == '2':
    train_model()
elif choice == '3':
    detect()
else:
    print("Invalid Choice")

cap.release()
cv2.destroyAllWindows()

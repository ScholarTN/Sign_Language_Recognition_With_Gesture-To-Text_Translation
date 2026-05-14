import cv2
import numpy as np
import joblib
import time

# Load trained model
model = joblib.load("asl_model.pkl")

IMG_SIZE = 64

# Webcam
cap = cv2.VideoCapture(0)

# Window setup
cv2.namedWindow("ASL Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ASL Recognition", 1000, 700)
cv2.moveWindow("ASL Recognition", 300, 100)

prediction = ""
last_prediction_time = 0

print("Press Q to quit")

while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror
    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape

    # ROI
    x1 = int(W * 0.55)
    y1 = int(H * 0.2)

    x2 = int(W * 0.95)
    y2 = int(H * 0.8)

    roi = frame[y1:y2, x1:x2]

    # Draw ROI box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Preprocess
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (35, 35), 0)

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    hand_detected = False

    if contours:

        max_contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(max_contour)

        # Require large contour
        if area > 12000:

            hand_detected = True

            x, y, w_box, h_box = cv2.boundingRect(max_contour)

            # Draw hand box
            cv2.rectangle(
                roi,
                (x, y),
                (x + w_box, y + h_box),
                (255, 0, 0),
                3
            )

            # Prepare ROI for model
            hand_img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

            features = hand_img.flatten().reshape(1, -1)

            # Predict slowly
            current_time = time.time()

            if current_time - last_prediction_time > 1:

                prediction = model.predict(features)[0]

                last_prediction_time = current_time

    # UI
    cv2.rectangle(frame, (10, 10), (500, 130), (0, 0, 0), -1)

    if hand_detected:

        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "No hand detected",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Instructions
    cv2.putText(
        frame,
        "Place hand inside green box",
        (500, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Show
    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
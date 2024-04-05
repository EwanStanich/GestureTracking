# Shows error but imports correctly
import cv2
import mediapipe as mp
import numpy as np
import uuid
import os


def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [1920,1080]).astype(int))
            
            output = text, coords
            
    return output

def draw_finger_angles(image, results, joint_list):
    
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle
                
            cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [1920, 1080]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return image

# Hand Track Set Up
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
joint_list = [[8,7,6], [12,11,10], [16,15,14], [20,19,18]]

# Webcam Set Up
cap = cv2.VideoCapture(0)
count = 0
timer = 3
is_taking_photo = False
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=4) as hands: 
    while cap.isOpened():

        if is_taking_photo:
            if count % 15 == 0:
                print(timer)
                timer -= 1
            count += 1
            if count % 45 == 0:
                print("Snap")
                cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
                is_taking_photo = False
                timer = 3


        ret, frame = cap.read()

        # Detections
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # print(results)

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=10),
                                        mp_drawing.DrawingSpec(color=(0, 250, 0), thickness=4, circle_radius=2),
                                         )
                # Render left or right detection
                if get_label(num, hand, results):
                    text, coord = get_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
            
            # Draw angles to image from joint list
            draw_finger_angles(image, results, joint_list)

        cv2.imshow("Hand Tracking", image)

        # Save  image    
        if cv2.waitKey(10) & 0xFF == ord('s'):
            is_taking_photo = True

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows
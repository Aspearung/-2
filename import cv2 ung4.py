import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 각도를 계산하는 함수
def calculate_angle(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    radians = math.atan2(dy, dx)
    angle = math.degrees(radians)
    return angle

# 시선 방향을 추정하는 함수
def estimate_gaze(nose, left_shoulder, right_shoulder):
    # 어깨의 중앙점을 계산합니다.
    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2

    # 코와 어깨 중앙 사이의 벡터를 계산합니다.
    direction_vector = [nose.x - shoulder_center_x, nose.y - shoulder_center_y]

    # 벡터의 각도를 계산합니다.
    angle = calculate_angle((shoulder_center_x, shoulder_center_y), (nose.x, nose.y))
    return angle

# 시선 방향을 출력하는 함수
def print_gaze_direction(angle):
    if -10 <= angle <= 10:  # 정면을 보고 있는 경우
        print("Gaze angle: {:.2f} - Forward".format(angle))
    elif angle < -10:  # 왼쪽을 보고 있는 경우
        print("Gaze angle: {:.2f} - Left".format(angle))
    else:  # 오른쪽을 보고 있는 경우
        print("Gaze angle: {:.2f} - Right".format(angle))

# 웹캠, 영상 파일의 경우 이것을 사용하세요.:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            continue

        # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # 포즈 주석을 이미지 위에 그립니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 코와 어깨 랜드마크를 가져옵니다.
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # 시선 방향을 추정합니다.
            angle = estimate_gaze(nose, left_shoulder, right_shoulder)
            print_gaze_direction(angle)

        # 보기 편하게 이미지를 좌우 반전합니다.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()

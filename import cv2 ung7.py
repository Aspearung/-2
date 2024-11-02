import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# 각도를 계산하는 함수
def calculate_angle(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    radians = math.atan2(dy, dx)
    angle = math.degrees(radians)
    return angle

# 시선 방향을 출력하는 함수
def print_gaze_direction(nose, left_eye, right_eye):
    # 눈 중앙점을 계산합니다.
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_center_y = (left_eye[1] + right_eye[1]) / 2

    # 코와 눈 중앙 사이의 벡터를 계산합니다.
    direction_vector = [nose[0] - eye_center_x, nose[1] - eye_center_y]

    # 벡터의 각도를 계산합니다.
    angle = calculate_angle((eye_center_x, eye_center_y), nose)

    # 코와 눈의 위치에 따라 시선 방향을 결정합니다.
    if angle < 0:  # 아래를 보는 경우
        return "Down"
    elif angle > 100:  # 위를 보는 경우
        return "Up"
    else:  # 정면을 보거나 측정 불가능한 경우
        return "Forward"

# 웹캠, 영상 파일의 경우 이것을 사용하세요.:
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            continue

        # 이미지를 RGB로 변환합니다.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 이미지 내에서 얼굴 랜드마크를 찾습니다.
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            # 얼굴 랜드마크를 추출합니다.
            face_landmarks = results.multi_face_landmarks[0].landmark

            # 코의 위치를 추정합니다.
            nose = [face_landmarks[1].x, face_landmarks[1].y]  # 얼굴 중앙의 첫 번째 점이 코의 근사 위치입니다.

            # 눈의 위치를 가져옵니다.
            left_eye = [face_landmarks[159].x, face_landmarks[159].y]  # 왼쪽 눈의 중앙
            right_eye = [face_landmarks[386].x, face_landmarks[386].y]  # 오른쪽 눈의 중앙

            # 시선 방향을 출력합니다.
            gaze_direction = print_gaze_direction(nose, left_eye, right_eye)
            print("Gaze direction:", gaze_direction)

            # 포즈 주석을 이미지 위에 그립니다.
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.multi_face_landmarks[0],
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

            # 이미지를 표시합니다.
            cv2.imshow('MediaPipe Face Mesh', annotated_image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()

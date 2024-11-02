import cv2
import tkinter as tk
from tkinter import messagebox, filedialog
import os
import threading
import time
import datetime
import matplotlib.pyplot as plt
from openpyxl import Workbook
import numpy as np
import mediapipe as mp
import math
from openpyxl import Workbook
from easygui import buttonbox
from datetime import datetime

# 전역 변수 설정
analysis_thread = None
running = False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 수업 분석 중인지 상태를 저장하는 플래그
is_analyzing = False

# 메인 수업 분석 함수 (사용자 제공 코드의 메인 내용 포함)
def analyze_class():
    global is_analyzing
    is_analyzing = True

    # 수업 분석 코드를 여기에 넣으세요.
    # 아래는 예제입니다.
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # 각 순서쌍을 카운트하는 딕셔너리
    counts = {
        'up_right': 0, 'up_left': 0, 'up_forward': 0,
        'neutral_right': 0, 'neutral_left': 0, 'neutral_forward': 0,
        'down_right': 0, 'down_left': 0, 'down_forward': 0
    }

    def calculate_angle(point1, point2):
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        radians = math.atan2(dy, dx)
        angle = math.degrees(radians)
        return angle

    def estimate_gaze(nose, left_shoulder, right_shoulder):
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        direction_vector = [nose.x - shoulder_center_x, nose.y - shoulder_center_y]
        angle = calculate_angle((shoulder_center_x, shoulder_center_y), (nose.x, nose.y))
        if angle < 0:
            angle += 360
        return angle, direction_vector

    def get_gaze_direction(angle, direction_vector):
        vertical_threshold_up = 0.25  # 'Up' 감지 임계값 (up 잘 감지됨)
        vertical_threshold_down = -0.1  # 'Down' 감지 임계값을 up과 보수 관계로 조정
        horizontal_threshold = 0.1

        vertical_direction = "Neutral"
        horizontal_direction = "Neutral"

        if direction_vector[1] < -vertical_threshold_up:  # 'Up' 방향 감지
            vertical_direction = "Up"
        elif direction_vector[1] > vertical_threshold_down:  # 'Down' 방향 감지
            vertical_direction = "Down"

        if 260 <= angle <= 280:
            horizontal_direction = "Forward"
        elif angle < 260:
            horizontal_direction = "Right"
        else:
            horizontal_direction = "Left"

        return vertical_direction, horizontal_direction

    # 웹캠
    cap = cv2.VideoCapture(0)

    time_intervals = []  # 시간을 저장할 리스트
    current_combination = None  # 현재 방향 조합
    start_time = time.time()  # 시작 시간 기록
    last_time = start_time  # 마지막으로 기록된 시간

    # 시간과 조합을 기록할 리스트
    combinations = []
    x_values = []
    y_values = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("카메라를 찾을 수 없습니다.")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                nose = landmarks[mp_pose.PoseLandmark.NOSE]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                angle, direction_vector = estimate_gaze(nose, left_shoulder, right_shoulder)
                vertical_direction, horizontal_direction = get_gaze_direction(angle, direction_vector)

                # 순서쌍 생성
                combination = f"{vertical_direction.lower()}_{horizontal_direction.lower()}"

                # 현재 조합이 변경되었을 때
                if combination != current_combination:
                    if current_combination is not None:
                        # 이전 조합의 시간과 조합을 기록
                        duration = time.time() - last_time
                        x_values.extend([last_time - start_time] * int(duration * 10))  # 0.1초 간격으로 표시
                        y_values.extend([current_combination] * int(duration * 10))  # 조합 저장
                    current_combination = combination
                    last_time = time.time()  # 현재 시간 기록
                    counts[combination] += 1  # 현재 조합 카운트 증가
                    print(f"Gaze direction: {vertical_direction}, {horizontal_direction}")

            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            cv2.imshow('MediaPipe Pose', annotated_image)

            if cv2.waitKey(5) & 0xFF == 27:  # ESC 키를 누르면 종료
                break

    # 마지막 조합도 기록
    if current_combination is not None:
        duration = time.time() - last_time
        x_values.extend([last_time - start_time] * int(duration * 10))
        y_values.extend([current_combination] * int(duration * 10))

    cap.release()
    cv2.destroyAllWindows()

    # 프로그램 종료 시 감지된 횟수를 출력
    print("Counts by combinations:")
    for key, value in counts.items():
        print(f"{key}: {value}")

    # 그래프 그리기
    plt.figure(figsize=(12, 6))

    # 고유 조합을 찾아서 색상 매핑
    unique_combinations = list(set(y_values))
    color_map = {comb: plt.cm.jet(i / len(unique_combinations)) for i, comb in enumerate(unique_combinations)}

    # 조합에 따른 영역 표시
    for comb in unique_combinations:
        comb_indices = [i for i, x in enumerate(y_values) if x == comb]
        plt.fill_between(
            [x_values[i] for i in comb_indices],
            [0] * len(comb_indices),
            [1] * len(comb_indices),  # y값은 임의로 설정, 구간 표시만 위해 사용
            color=color_map[comb],
            alpha=0.5,
            label=comb.replace('_', ' ').title()
        )

    # 꺾은선 그래프 그리기
    plt.plot(x_values, [unique_combinations.index(y) for y in y_values], color='black', linewidth=2)

    plt.title('Gaze Direction Combinations Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Gaze Directions')
    plt.yticks(range(len(unique_combinations)), unique_combinations)
    plt.legend()
    plt.grid()

    # 원 그래프 그리기
    plt.figure(figsize=(6, 6))
    plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=90)
    plt.title('Gaze Direction Distribution')
    plt.axis('equal')  # 원형으로 만들기 위해

    # 그래프 동시에 표시
    plt.show()

    # 사용자에게 저장 여부 확인
    choices = ["Yes", "No"]
    save_data = buttonbox("데이터를 저장하시겠습니까?", choices=choices)

    if save_data == "Yes":
        # 현재 시간을 이용한 파일 이름 생성
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Excel 파일로 저장
        wb = Workbook()
        ws = wb.active
        ws.append(['Combination', 'Count'])

        for key, value in counts.items():
            ws.append([key, value])

        excel_file = os.path.join(os.getcwd(), f'gaze_direction_counts_{current_time}.xlsx')
        wb.save(excel_file)
        print(f"Excel 파일이 {excel_file} 경로에 저장되었습니다.")

        # 그래프 이미지로 저장
        graph_file = os.path.join(os.getcwd(), f'gaze_direction_combinations_{current_time}.png')
        plt.savefig(graph_file)
        print(f"그래프 이미지가 {graph_file} 경로에 저장되었습니다.")

        cap.release()
        cv2.destroyAllWindows()
        is_analyzing = False
    # 카메라와 파일 저장 종료
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # 엑셀 파일에 데이터 저장
        wb = Workbook()
        ws = wb.active
        ws.append(["Combination", "Count"])
        for key, value in counts.items():
            ws.append([key, value])
        wb.save(excel_filename)

        # 분석 그래프 이미지 저장
        plt.figure(figsize=(6, 6))
        plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=90)
        plt.title("Gaze Direction Distribution")
        plt.axis("equal")
        plt.savefig(image_filename)
        plt.close()
        messagebox.showinfo("완료", f"수업 분석이 완료되었습니다.\n동영상, 이미지, 엑셀 파일이 저장되었습니다.")

# 분석을 시작하는 함수
def start_analysis():
    global analysis_thread, running
    if analysis_thread and analysis_thread.is_alive():
        messagebox.showwarning("경고", "분석이 이미 진행 중입니다.")
        return
    analysis_thread = threading.Thread(target=analyze_class())
    analysis_thread.start()

# 분석을 종료하는 함수
def stop_analysis():
    global running
    running = False
    messagebox.showinfo("종료", "수업 분석이 종료되었습니다.")

# 기록된 파일 보기 및 열기
def view_recordings():
    files = [f for f in os.listdir(recordings_dir) if os.path.isfile(os.path.join(recordings_dir, f))]
    if not files:
        messagebox.showinfo("알림", "저장된 기록이 없습니다.")
        return

    selected_file = filedialog.askopenfilename(initialdir=recordings_dir, title="파일을 선택하세요")
    if selected_file:
        if selected_file.endswith(".avi"):
            os.startfile(selected_file)  # 영상 파일 열기
        elif selected_file.endswith(".png"):
            img = plt.imread(selected_file)
            plt.imshow(img)
            plt.axis("off")
            plt.show()  # 이미지 파일 열기
        elif selected_file.endswith(".xlsx"):
            os.startfile(selected_file)  # 엑셀 파일 열기

# 도움말
def show_help():
    messagebox.showinfo("도움말", "안녕하세요")

# 프로그램 종료
def exit_program():
    global running
    running = False
    root.destroy()

# 메인 윈도우 설정
root = tk.Tk()
root.title("수업 분석 프로그램")
root.geometry("600x400")
root.configure(bg="lightblue")

# 메뉴 레이블과 버튼 추가
label = tk.Label(root, text="수업 분석 프로그램", font=("Arial", 20), bg="lightblue")
label.pack(pady=20)

button1 = tk.Button(root, text="1. 수업 분석 시작하기", command=start_analysis, font=("Arial", 14), width=25)
button1.pack(pady=10)

button5 = tk.Button(root, text="1-1. 수업 분석 종료하기", command=stop_analysis, font=("Arial", 14), width=25)
button5.pack(pady=10)

button2 = tk.Button(root, text="2. 나의 수업 돌아보기", command=view_recordings, font=("Arial", 14), width=25)
button2.pack(pady=10)

button3 = tk.Button(root, text="3. 도움말", command=show_help, font=("Arial", 14), width=25)
button3.pack(pady=10)

button4 = tk.Button(root, text="4. 종료하기", command=exit_program, font=("Arial", 14), width=25)
button4.pack(pady=10)

root.mainloop()

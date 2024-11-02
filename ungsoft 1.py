import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from datetime import datetime

# 수업 분석을 위한 주요 함수들 정의
def main_program():
    # 여기에 사용자가 제공한 시선 감지 코드를 삽입하여 실행하도록 합니다.
    pass

def show_graph():
    # 예제 그래프 생성 (사용자 제공 데이터로 대체 가능)
    counts = {
        'up_right': 10, 'up_left': 20, 'up_forward': 15,
        'neutral_right': 25, 'neutral_left': 30, 'neutral_forward': 10,
        'down_right': 5, 'down_left': 10, 'down_forward': 20
    }
    plt.figure(figsize=(6, 6))
    plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=90)
    plt.title('Gaze Direction Distribution')
    plt.axis('equal')
    plt.show()

# GUI 설정
def start_analysis():
    main_program()

def view_class():
    show_graph()

def show_help():
    messagebox.showinfo("도움말", "안녕하세요")

def exit_program():
    root.destroy()

# 메인 윈도우 설정
root = tk.Tk()
root.title("수업 분석 프로그램")
root.geometry("600x400")  # 창 크기를 600x400으로 설정
root.configure(bg="lightblue")  # 창 배경을 파란색으로 설정

# 메뉴 레이블과 버튼 추가
label = tk.Label(root, text="수업 분석 프로그램", font=("Arial", 20), bg="lightblue")
label.pack(pady=20)

button1 = tk.Button(root, text="1. 수업 분석 시작하기", command=start_analysis, font=("Arial", 14), width=25)
button1.pack(pady=10)

button2 = tk.Button(root, text="2. 나의 수업 돌아보기", command=view_class, font=("Arial", 14), width=25)
button2.pack(pady=10)

button3 = tk.Button(root, text="3. 도움말", command=show_help, font=("Arial", 14), width=25)
button3.pack(pady=10)

button4 = tk.Button(root, text="4. 종료하기", command=exit_program, font=("Arial", 14), width=25)
button4.pack(pady=10)

root.mainloop()

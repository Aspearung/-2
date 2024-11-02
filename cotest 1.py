import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

# 시선 분석 메인 코드 함수로 정의
def main_program():
    pass

def review_classes():
    pass

def change_bg_color():
    color = color_var.get()
    window.config(bg=color)

# 메인 윈도우 설정
def create_main_window():
    global window, color_var
    window = tk.Tk()
    window.title("수업 분석 프로그램")
    window.geometry("800x600")  # 창 크기 설정
    window.config(bg="lightgray")  # 기본 배경색 설정

    color_var = tk.StringVar(value="lightgray")

    # 메뉴 버튼 설정
    def start_analysis():
        main_program()

    def review():
        review_classes()

    def show_help():
        messagebox.showinfo("도움말", "안녕하세요")

    def exit_program():
        window.quit()

    font_settings = ("Helvetica", 20)

    btn1 = ttk.Button(window, text="수업 분석 시작하기", command=start_analysis, width=20)
    btn1.pack(pady=10)

    btn2 = ttk.Button(window, text="나의 수업 돌아보기", command=review, width=20)
    btn2.pack(pady=10)

    btn3 = ttk.Button(window, text="도움말", command=show_help, width=20)
    btn3.pack(pady=10)

    btn4 = ttk.Button(window, text="종료하기", command=exit_program, width=20)
    btn4.pack(pady=10)

    # 배경색 변경 옵션 추가
    color_label = ttk.Label(window, text="배경색 변경:", font=font_settings)
    color_label.pack(pady=10)
    color_option = ttk.OptionMenu(window, color_var, "lightgray", "white", "lightblue", "lightgreen", command=lambda _: change_bg_color())
    color_option.pack(pady=10)

    window.mainloop()

# 메인 프로그램 실행
if __name__ == "__main__":
    create_main_window()

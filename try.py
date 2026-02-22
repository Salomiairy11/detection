# Human Detection System

from tkinter import *
import tkinter as tk
import tkinter.messagebox as mbox
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import argparse
from persondetection import DetectorAPI
import matplotlib.pyplot as plt
from fpdf import FPDF
import os


# ---------- GLOBAL VARIABLES ----------
exit1 = False
filename = ""
filename1 = ""
filename2 = ""
odapi = DetectorAPI()



# ---------- ARGUMENT PARSER ----------
def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File")
    arg_parse.add_argument("-i", "--image", default=None, help="path to Image File")
    arg_parse.add_argument("-c", "--camera", default=False, help="Use camera")
    arg_parse.add_argument("-o", "--output", type=str, help="output video path")
    return vars(arg_parse.parse_args())


# ---------- SPLASH WINDOW ----------
window = tk.Tk()
window.title("Human Detection System")
window.iconbitmap('Images/icon.ico')
window.geometry('1000x700')
window.resizable(False, False)

# ---------- BACKGROUND IMAGE ----------
bg_img = ImageTk.PhotoImage(Image.open("Images/front1.PNG").resize((1000, 700)))
bg_label = tk.Label(window, image=bg_img)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# ---------- TITLE ----------
title_label = tk.Label(
    window,
    text="HUMAN DETECTION SYSTEM",
    font=("Segoe UI", 36, "bold"),
    fg="#0A1F44",      # dark navy
    bg="#ffffff"
)
title_label.place(relx=0.5, rely=0.15, anchor="center")

# ---------- FUNCTIONS ----------
def start_fun():
    window.destroy()

def exit_win():
    global exit1
    if mbox.askokcancel("Exit", "Do you want to exit?"):
        exit1 = True
        window.destroy()

# ---------- BUTTON STYLE ----------
btn_style = {
    "font": ("Segoe UI", 18, "bold"),
    "width": 15,
    "height": 1,
    "bd": 0,
    "cursor": "hand2"
}

# ---------- START BUTTON ----------
start_btn = tk.Button(
    window,
    text="START",
    command=start_fun,
    bg="#1F3A5F",   # formal dark blue
    fg="white",
    activebackground="#162B47",
    **btn_style
)
start_btn.place(relx=0.5, rely=0.55, anchor="center")

# ---------- EXIT BUTTON ----------
exit_btn = tk.Button(
    window,
    text="EXIT",
    command=exit_win,
    bg="#6B0F1A",   # formal dark red
    fg="white",
    activebackground="#4A0A12",
    **btn_style
)
exit_btn.place(relx=0.5, rely=0.67, anchor="center")

window.protocol("WM_DELETE_WINDOW", exit_win)
window.mainloop()


# ---------- MAIN SYSTEM WINDOW ----------
if not exit1:
    window1 = tk.Tk()
    window1.title("Human Detection System")
    window1.iconbitmap('Images/icon.ico')
    window1.geometry('1000x700')

# ---------------------------- image section ------------------------------------------------------------
def image_option():
    import tkinter as tk
    from tkinter import filedialog, messagebox as mbox
    import cv2
    from fpdf import FPDF
    import matplotlib.pyplot as plt

    # ---------- Window ----------
    windowi = tk.Toplevel(window1)
    windowi.title("Human Detection from Image")
    windowi.iconbitmap('Images/icon.ico')
    windowi.geometry('1000x700')
    windowi.configure(bg="#f5f5f5")

    # ---------- State Variables ----------
    filename1 = ""
    max_count1 = 0
    framex1 = []
    county1 = []
    max1 = []
    avg_acc1_list = []
    max_avg_acc1_list = []
    max_acc1 = 0
    max_avg_acc1 = 0

    # ---------- Frames ----------
    frame_top = tk.Frame(windowi, bg="#f5f5f5")
    frame_top.pack(pady=10)

    frame_select = tk.LabelFrame(windowi, text="Image Selection", font=("Segoe UI", 14), padx=10, pady=10)
    frame_select.pack(fill="x", padx=20, pady=10)

    frame_action = tk.LabelFrame(windowi, text="Actions", font=("Segoe UI", 14), padx=10, pady=10)
    frame_action.pack(fill="x", padx=20, pady=10)

    frame_status = tk.Frame(windowi, bg="#f5f5f5")
    frame_status.pack(fill="x", padx=20, pady=10)

    # ---------- Title ----------
    lbl_title = tk.Label(frame_top, text="Human Detection from Image", font=("Segoe UI", 24, "bold"), bg="#f5f5f5", fg="#333")
    lbl_title.pack(pady=10)

    # ---------- Image Path ----------
    path_text1 = tk.Entry(frame_select, font=("Segoe UI", 12), width=60)
    path_text1.pack(side="left", padx=5, pady=5)

    def open_img():
        nonlocal filename1, max_count1, framex1, county1, max1, avg_acc1_list, max_avg_acc1_list, max_acc1, max_avg_acc1
        max_count1 = 0
        framex1.clear()
        county1.clear()
        max1.clear()
        avg_acc1_list.clear()
        max_avg_acc1_list.clear()
        max_acc1 = 0
        max_avg_acc1 = 0

        filename1 = filedialog.askopenfilename(title="Select Image file", filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
        path_text1.delete(0, "end")
        path_text1.insert(0, filename1)

    btn_select = tk.Button(frame_select, text="Select Image", command=open_img, font=("Segoe UI", 12), bg="#4caf50", fg="white")
    btn_select.pack(side="left", padx=5)

    # ---------- Preview ----------
    def prev_img():
        if not filename1:
            mbox.showerror("Error", "No Image File Selected!", parent=windowi)
            return
        img = cv2.imread(filename1)
        if img is None:
            mbox.showerror("Error", "Cannot open image.", parent=windowi)
            return
        cv2.imshow("Selected Image Preview", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    btn_preview = tk.Button(frame_select, text="Preview", command=prev_img, font=("Segoe UI", 12), bg="#2196f3", fg="white")
    btn_preview.pack(side="left", padx=5)

    # ---------- Detection ----------
    def detectByPathImage(path):
        nonlocal max_count1, max_acc1, max_avg_acc1, framex1, county1, max1, avg_acc1_list, max_avg_acc1_list

        threshold = 0.7

        image = cv2.imread(path)
        if image is None:
            mbox.showerror("Error", "Cannot open image.", parent=windowi)
            return

        img = cv2.resize(image, (image.shape[1], image.shape[0]))
        boxes, scores, classes, num = odapi.processFrame(img)

        person = 0
        acc = 0

        for i in range(len(boxes)):
            if classes[i] == 0 and scores[i] > threshold:
                box = boxes[i]
                person += 1
                acc += scores[i]
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
                cv2.putText(img, f'P{person, round(scores[i],2)}', (box[1]-30, box[0]-8), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1)
                if scores[i] > max_acc1:
                    max_acc1 = scores[i]

        max_count1 = person
        max_avg_acc1 = (acc/person) if person > 0 else 0

        cv2.imshow("Human Detection from Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        for i in range(20):
            framex1.append(i)
            county1.append(max_count1)
            max1.append(max_count1)
            avg_acc1_list.append(max_avg_acc1)
            max_avg_acc1_list.append(max_avg_acc1)

        info1.config(text="Status: Detection & Counting Completed")

        # Enable plot and report buttons after detection
        btn_enum.config(state="normal")
        btn_acc.config(state="normal")
        btn_report.config(state="normal")

    def det_img():
        if not filename1:
            mbox.showerror("Error", "No Image File Selected!", parent=windowi)
            return
        mbox.showinfo("Status", "Detecting, Please Wait...", parent=windowi)
        detectByPathImage(filename1)

    btn_detect = tk.Button(frame_action, text="Detect", command=det_img, font=("Segoe UI", 12), bg="#ff9800", fg="white")
    btn_detect.pack(side="left", padx=10, pady=10)

    # ---------- Plots ----------
    def img_enumeration_plot():
        plt.figure(facecolor='white')
        ax = plt.axes()
        ax.set_facecolor("#e0e0e0")
        plt.plot(framex1, county1, label="Human Count", marker='o', color="#2196f3")
        plt.plot(framex1, max1, linestyle='dashed', label="Max Count", color="#f44336")
        plt.xlabel("Time (sec)")
        plt.ylabel("Human Count")
        plt.legend()
        plt.title("Enumeration Plot")
        plt.show()

    def img_accuracy_plot():
        plt.figure(facecolor='white')
        ax = plt.axes()
        ax.set_facecolor("#e0e0e0")
        plt.plot(framex1, avg_acc1_list, label="Avg Accuracy", marker='o', color="#4caf50")
        plt.plot(framex1, max_avg_acc1_list, linestyle='dashed', label="Max Avg Accuracy", color="#f44336")
        plt.xlabel("Time (sec)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy Plot")
        plt.show()

    btn_enum = tk.Button(frame_action, text="Enumeration Plot", command=img_enumeration_plot, state="disabled", font=("Segoe UI", 12), bg="#ff9800", fg="white")
    btn_enum.pack(side="left", padx=10, pady=10)

    btn_acc = tk.Button(frame_action, text="Avg Accuracy Plot", command=img_accuracy_plot, state="disabled", font=("Segoe UI", 12), bg="#ff9800", fg="white")
    btn_acc.pack(side="left", padx=10, pady=10)

    # ---------- Report ----------
    # def img_gen_report():
    #     pdf = FPDF()
    #     pdf.add_page()
    #     pdf.set_font("Arial", "", 20)
    #     pdf.set_text_color(128,0,0)
    #     pdf.image('Images/Crowd_Report.png', x=0, y=0, w=210, h=297)

    #     pdf.text(125, 150, str(max_count1))
    #     pdf.text(105, 163, str(max_acc1))
    #     pdf.text(125, 175, str(max_avg_acc1))

    #     if max_count1 > 25:
    #         pdf.text(26, 220, "Max Human Detected is above MAX LIMIT.")
    #         pdf.text(70, 235, "Region is Crowded.")
    #     else:
    #         pdf.text(26, 220, "Max Human Detected is within MAX LIMIT.")
    #         pdf.text(65, 235, "Region is not Crowded.")

    #     pdf.output('Crowd_Report.pdf')
    #     mbox.showinfo("Status", "Report Generated Successfully.", parent=windowi)

    def img_gen_report():
        if not framex1:
            mbox.showwarning("No Data", "Run detection first.", parent=windowi)
            return

        pdf = FPDF()
        pdf.add_page()

        # ===== Title =====
        pdf.set_font("Arial", "B", 20)
        pdf.cell(0, 12, "Human Detection Analysis Report", ln=True, align="C")

        pdf.ln(5)

        # ===== Input Information =====
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Input Information", ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Source Type: Image", ln=True)
        pdf.cell(0, 8, f"Image Path: {filename1}", ln=True)
        pdf.cell(0, 8, f"Detection Threshold: 0.7", ln=True)

        pdf.ln(5)

        # ===== Compute Statistics =====
        total_frames = len(framex1)
        avg_count = sum(county1) / total_frames if total_frames else 0
        min_count = min(county1) if county1 else 0
        variation = max(county1) - min_count if county1 else 0
        confidence = (
            "High" if max_avg_acc1 > 0.85 else
            "Moderate" if max_avg_acc1 > 0.70 else
            "Low"
        )

        if max_count1 > 25:
            crowd_status = "High Density Crowd"
        elif max_count1 > 10:
            crowd_status = "Moderate Crowd"
        else:
            crowd_status = "Low Crowd Presence"

        # ===== Detection Summary Table =====
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Detection Summary", ln=True)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(95, 10, "Metric", border=1)
        pdf.cell(95, 10, "Value", border=1, ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.cell(95, 10, "Maximum Humans Detected", border=1)
        pdf.cell(95, 10, str(max_count1), border=1, ln=True)

        pdf.cell(95, 10, "Average Humans Detected", border=1)
        pdf.cell(95, 10, str(round(avg_count, 2)), border=1, ln=True)

        pdf.cell(95, 10, "Minimum Humans Detected", border=1)
        pdf.cell(95, 10, str(min_count), border=1, ln=True)

        pdf.cell(95, 10, "Maximum Detection Accuracy", border=1)
        pdf.cell(95, 10, str(round(max_acc1, 2)), border=1, ln=True)

        pdf.cell(95, 10, "Average Detection Accuracy", border=1)
        pdf.cell(95, 10, str(round(max_avg_acc1, 2)), border=1, ln=True)

        pdf.cell(95, 10, "Total Frames Analysed", border=1)
        pdf.cell(95, 10, str(total_frames), border=1, ln=True)

        pdf.ln(5)

        # ===== Statistical Analysis =====
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Statistical Analysis", ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Crowd Density Level: {crowd_status}", ln=True)
        pdf.cell(0, 8, f"Detection Confidence Level: {confidence}", ln=True)
        pdf.cell(0, 8, f"Crowd Variation Range: {variation}", ln=True)

        pdf.ln(5)

        # ===== Interpretation =====
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "System Interpretation", ln=True)

        pdf.set_font("Arial", "", 12)

        if max_count1 > 25:
            interpretation = "High crowd density detected. Monitoring recommended."
        elif max_count1 > 10:
            interpretation = "Moderate human presence detected."
        else:
            interpretation = "Low human presence detected."

        pdf.multi_cell(0, 8, interpretation)

        pdf.ln(5)

        # ===== System Information =====
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "System Information", ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, "Detection Model: TensorFlow Person Detection", ln=True)
        pdf.cell(0, 8, "Framework: OpenCV + TensorFlow", ln=True)
        pdf.cell(0, 8, "Processing Resolution: Original Image Size", ln=True)

        # ===== Save PDF =====
        report_path = os.path.abspath("Crowd_Report.pdf")

        try:
            if os.path.exists(report_path):
                os.remove(report_path)
        except:
            pass

        pdf.output(report_path)
        mbox.showinfo("Status", "Report Generated Successfully.", parent=windowi)

        os.startfile(report_path)
    
    btn_report = tk.Button(frame_action, text="Generate Report", command=img_gen_report, state="disabled", font=("Segoe UI", 12), bg="#ff9800", fg="white")
    btn_report.pack(side="left", padx=10, pady=10)

    # ---------- Status ----------
    info1 = tk.Label(frame_status, font=("Segoe UI", 12), bg="#f5f5f5", fg="#333")
    info1.pack(anchor="w")

    # ---------- Exit ----------
    def exit_wini():
        if mbox.askokcancel("Exit", "Do you want to exit?", parent=windowi):
            windowi.destroy()

    windowi.protocol("WM_DELETE_WINDOW", exit_wini)
    windowi.mainloop()


# ---------------------------- video section ------------------------------------------------------------
def video_option():
    global filename2
    filename2 = ""

    # ---------- Video Window ----------
    windowv = tk.Toplevel(window1)
    windowv.title("Human Detection from Video")
    windowv.iconbitmap('Images/icon.ico')
    windowv.geometry('1000x700')
    windowv.configure(bg="#f5f5f5")
    
    # ---------- State Variables ----------
    max_count2 = 0
    framex2 = []
    county2 = []
    max2 = []
    avg_acc2_list = []
    max_avg_acc2_list = []
    max_acc2 = 0
    max_avg_acc2 = 0

    # ---------- Frames ----------
    frame_top = tk.Frame(windowv, bg="#f5f5f5")
    frame_top.pack(pady=10)

    frame_select = tk.LabelFrame(windowv, text="Video Selection", font=("Segoe UI", 14), padx=10, pady=10)
    frame_select.pack(fill="x", padx=20, pady=10)

    frame_action = tk.LabelFrame(windowv, text="Actions", font=("Segoe UI", 14), padx=10, pady=10)
    frame_action.pack(fill="x", padx=20, pady=10)

    frame_status = tk.Frame(windowv, bg="#f5f5f5")
    frame_status.pack(fill="x", padx=20, pady=10)

    # ---------- Title ----------
    lbl_title = tk.Label(frame_top, text="Human Detection from Video", font=("Segoe UI", 24, "bold"), bg="#f5f5f5", fg="#333")
    lbl_title.pack(pady=10)

    # ---------- Video Path Entry ----------
    path_text2 = tk.Entry(frame_select, font=("Segoe UI", 12), width=60)
    path_text2.pack(side="left", padx=5, pady=5)

    def open_vid():
        nonlocal max_count2, framex2, county2, max2, avg_acc2_list, max_avg_acc2_list, max_acc2, max_avg_acc2
        global filename2
        max_count2 = 0
        framex2.clear()
        county2.clear()
        max2.clear()
        avg_acc2_list.clear()
        max_avg_acc2_list.clear()
        max_acc2 = 0
        max_avg_acc2 = 0

        filename2 = filedialog.askopenfilename(title="Select Video file", filetypes=[("Video files", "*.mp4 *.avi")])
        path_text2.delete(0, "end")
        path_text2.insert(0, filename2)

    btn_select = tk.Button(frame_select, text="Select Video", command=open_vid, font=("Segoe UI", 12), bg="#4caf50", fg="white")
    btn_select.pack(side="left", padx=5)

    # ---------- Preview ----------
    def prev_vid():
        if filename2 == "":
            mbox.showerror("Error", "No Video File Selected!", parent=windowv)
            return

        cap = cv2.VideoCapture(filename2)
        if not cap.isOpened():
            mbox.showerror("Error", "Cannot open video.", parent=windowv)
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.resize(frame, (800, 500))
            cv2.imshow('Video Preview', img)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    btn_preview = tk.Button(frame_select, text="Preview", command=prev_vid, font=("Segoe UI", 12), bg="#2196f3", fg="white")
    btn_preview.pack(side="left", padx=10)

    # ---------- Detection ----------
    def detectByPathVideo(path, writer=None):
        nonlocal max_count2, framex2, county2, max2, avg_acc2_list, max_avg_acc2_list, max_acc2, max_avg_acc2

        video = cv2.VideoCapture(path)
        threshold = 0.7

        if not video.isOpened():
            mbox.showerror("Error", "Video not found.", parent=windowv)
            return

        frame_index = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            img = cv2.resize(frame, (800, 500))
            boxes, scores, classes, num = odapi.processFrame(img)

            person = 0
            acc = 0

            for i in range(len(boxes)):
                if classes[i] == 0 and scores[i] > threshold:
                    box = boxes[i]
                    person += 1
                    acc += scores[i]

                    cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
                    cv2.putText(img, f'P{person, round(scores[i], 2)}', (box[1]-30, box[0]-8),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

                    if scores[i] > max_acc2:
                        max_acc2 = scores[i]

            if person > max_count2:
                max_count2 = person

            if person > 0:
                avg = acc / person
                avg_acc2_list.append(avg)
                if avg > max_avg_acc2:
                    max_avg_acc2 = avg
            else:
                avg_acc2_list.append(0)

            frame_index += 1
            framex2.append(frame_index)
            county2.append(person)

            if writer is not None:
                writer.write(img)

            cv2.imshow("Human Detection from Video", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

        for _ in framex2:
            max2.append(max_count2)
            max_avg_acc2_list.append(max_avg_acc2)

        info1.config(text="Status: Detection & Counting Completed")

        # Enable buttons after detection
        btn_enum.config(state="normal")
        btn_acc.config(state="normal")
        btn_report.config(state="normal")

    def det_vid():
        if filename2 == "":
            mbox.showerror("Error", "No Video File Selected!", parent=windowv)
            return

        mbox.showinfo("Status", "Detecting, Please Wait...", parent=windowv)
        args = argsParser()
        writer = None
        if args['output'] is not None:
            writer = cv2.VideoWriter(args['output'],
                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                     10, (800, 500))

        detectByPathVideo(filename2, writer)

    btn_detect = tk.Button(frame_action, text="Detect", command=det_vid, font=("Segoe UI", 12), bg="#ff9800", fg="white")
    btn_detect.pack(side="left", padx=18, pady=18)

    # ---------- Plots ----------
    def vid_enumeration_plot():
        plt.figure(facecolor='white')
        ax = plt.axes()
        ax.set_facecolor("#e0e0e0")
        plt.plot(framex2, county2, label="Human Count", marker='o', color="#2196f3")
        plt.plot(framex2, max2, linestyle='dashed', label="Max Human Count", color="#f44336")
        plt.xlabel('Frame')
        plt.ylabel('Human Count')
        plt.legend()
        plt.title('Enumeration Plot')
        plt.show()

    def vid_accuracy_plot():
        plt.figure(facecolor='white')
        ax = plt.axes()
        ax.set_facecolor("#e0e0e0")
        plt.plot(framex2, avg_acc2_list, label="Avg Accuracy", marker='o', color="#4caf50")
        plt.plot(framex2, max_avg_acc2_list, linestyle='dashed', label="Max Avg Accuracy", color="#f44336")
        plt.xlabel('Frame')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Plot')
        plt.show()

    btn_enum = tk.Button(frame_action, text="Enumeration Plot", command=vid_enumeration_plot, state="disabled", font=("Segoe UI", 12), bg="#ff9800", fg="white")
    btn_enum.pack(side="left", padx=18, pady=18)

    btn_acc = tk.Button(frame_action, text="Avg Accuracy Plot", command=vid_accuracy_plot, state="disabled", font=("Segoe UI", 12), bg="#ff9800", fg="white")
    btn_acc.pack(side="left", padx=18, pady=18)

    # ---------- Report ----------
    # def vid_gen_report():
    #     pdf = FPDF()
    #     pdf.add_page()
    #     pdf.set_font("Arial", "", 20)
    #     pdf.set_text_color(128, 0, 0)
    #     pdf.image('Images/Crowd_Report.png', x=0, y=0, w=210, h=297)

    #     pdf.text(125, 150, str(max_count2))
    #     pdf.text(105, 163, str(max_acc2))
    #     pdf.text(125, 175, str(max_avg_acc2))

    #     if max_count2 > 25:
    #         pdf.text(26, 220, "Region is Crowded.")
    #     else:
    #         pdf.text(26, 220, "Region is not Crowded.")

    #     pdf.output('Crowd_Report.pdf')
    #     mbox.showinfo("Status", "Report Generated Successfully.", parent=windowv)
    
    def vid_gen_report():
        if not framex2:
            mbox.showwarning("No Data", "Run detection first.", parent=windowv)
            return

        pdf = FPDF()
        pdf.add_page()

        # ===== Title =====
        pdf.set_font("Arial", "B", 20)
        pdf.cell(0, 12, "Human Detection Analysis Report", ln=True, align="C")

        pdf.ln(5)

        # ===== Input Information =====
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Input Information", ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Source Type: Video", ln=True)
        pdf.cell(0, 8, f"Video Path: {filename2}", ln=True)
        pdf.cell(0, 8, f"Detection Threshold: 0.7", ln=True)

        pdf.ln(5)

        # ===== Compute Statistics =====
        total_frames = len(framex2)
        avg_count = sum(county2) / total_frames if total_frames else 0
        min_count = min(county2) if county2 else 0
        variation = max(county2) - min_count if county2 else 0
        confidence = (
            "High" if max_avg_acc2 > 0.85 else
            "Moderate" if max_avg_acc2 > 0.70 else
            "Low"
        )

        if max_count2 > 25:
            crowd_status = "High Density Crowd"
        elif max_count2 > 10:
            crowd_status = "Moderate Crowd"
        else:
            crowd_status = "Low Crowd Presence"

        # ===== Detection Summary Table =====
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Detection Summary", ln=True)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(95, 10, "Metric", border=1)
        pdf.cell(95, 10, "Value", border=1, ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.cell(95, 10, "Maximum Humans Detected", border=1)
        pdf.cell(95, 10, str(max_count2), border=1, ln=True)

        pdf.cell(95, 10, "Average Humans Detected", border=1)
        pdf.cell(95, 10, str(round(avg_count, 2)), border=1, ln=True)

        pdf.cell(95, 10, "Minimum Humans Detected", border=1)
        pdf.cell(95, 10, str(min_count), border=1, ln=True)

        pdf.cell(95, 10, "Maximum Detection Accuracy", border=1)
        pdf.cell(95, 10, str(round(max_acc2, 2)), border=1, ln=True)

        pdf.cell(95, 10, "Average Detection Accuracy", border=1)
        pdf.cell(95, 10, str(round(max_avg_acc2, 2)), border=1, ln=True)

        pdf.cell(95, 10, "Total Frames Analysed", border=1)
        pdf.cell(95, 10, str(total_frames), border=1, ln=True)

        pdf.ln(5)

        # ===== Statistical Analysis =====
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Statistical Analysis", ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Crowd Density Level: {crowd_status}", ln=True)
        pdf.cell(0, 8, f"Detection Confidence Level: {confidence}", ln=True)
        pdf.cell(0, 8, f"Crowd Variation Range: {variation}", ln=True)

        # ===== Save + Open =====
        report_path = os.path.abspath("Crowd_Report.pdf")

        try:
            if os.path.exists(report_path):
                os.remove(report_path)
        except:
            pass

        pdf.output(report_path)
        mbox.showinfo("Status", "Report Generated Successfully.", parent=windowv)
        os.startfile(report_path)

    btn_report = tk.Button(frame_action, text="Generate Report", command=vid_gen_report, state="disabled", font=("Segoe UI", 12), bg="#ff9800", fg="white")
    btn_report.pack(side="left", padx=18, pady=18)

    # ---------- Status ----------
    info1 = tk.Label(frame_status, font=("Segoe UI", 12), bg="#f5f5f5", fg="#333")
    info1.pack(anchor="w")

    # ---------- Exit ----------
    def exit_winv():
        if mbox.askokcancel("Exit", "Do you want to exit?", parent=windowv):
            windowv.destroy()

    windowv.protocol("WM_DELETE_WINDOW", exit_winv)
    windowv.mainloop()

# ---------------------------- camera section ------------------------------------------------------------
import threading

def camera_option():
    import tkinter as tk
    from tkinter import messagebox as mbox
    import cv2
    from fpdf import FPDF
    import matplotlib.pyplot as plt

    # ---------- Camera Window ----------
    global windowc
    windowc = tk.Toplevel(window1)
    windowc.title("Human Detection from Camera")
    windowc.iconbitmap('Images/icon.ico')
    windowc.geometry('1000x700')
    windowc.configure(bg="#f5f5f5")

    # ---------- State Variables ----------
    max_count3 = 0
    framex3 = []
    county3 = []
    max3 = []
    avg_acc3_list = []
    max_avg_acc3_list = []
    max_acc3 = 0
    max_avg_acc3 = 0

    # ---------- Frames ----------
    frame_top = tk.Frame(windowc, bg="#f5f5f5")
    frame_top.pack(pady=10)

    frame_action = tk.LabelFrame(windowc, text="Actions", font=("Segoe UI", 14), padx=10, pady=10)
    frame_action.pack(fill="x", padx=20, pady=10)

    frame_status = tk.Frame(windowc, bg="#f5f5f5")
    frame_status.pack(fill="x", padx=20, pady=10)

    # ---------- Title ----------
    lbl_title = tk.Label(frame_top, text="Human Detection from Camera", font=("Segoe UI", 24, "bold"), bg="#f5f5f5", fg="#333")
    lbl_title.pack(pady=10)

    # ---------- Status ----------
    info1 = tk.Label(frame_status, font=("Segoe UI", 12), bg="#f5f5f5", fg="#333")
    info1.pack(anchor="w")

    # ---------- Camera Detection Logic ----------
    def detectByCamera(writer=None):
        nonlocal max_count3, framex3, county3, max3, avg_acc3_list, max_avg_acc3_list, max_acc3, max_avg_acc3

        video = cv2.VideoCapture(0)
        threshold = 0.7
        x3 = 0

        if not video.isOpened():
            mbox.showerror("Error", "Cannot access camera!", parent=windowc)
            return

        info1.config(text="Status: Camera Running... Press 'q' to stop.")
        while True:
            check, frame = video.read()
            if not check:
                break

            img = cv2.resize(frame, (800, 600))
            boxes, scores, classes, num = odapi.processFrame(img)

            person = 0
            acc = 0

            for i in range(len(boxes)):
                if classes[i] == 0 and scores[i] > threshold:
                    box = boxes[i]
                    person += 1
                    acc += scores[i]
                    cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
                    cv2.putText(img, f'P{person, round(scores[i],2)}', (box[1]-30, box[0]-8), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255),1)
                    if scores[i] > max_acc3:
                        max_acc3 = scores[i]

            if person > max_count3:
                max_count3 = person

            county3.append(person)
            x3 += 1
            framex3.append(x3)

            avg_acc3_list.append(acc/person if person>0 else 0)
            if person>0 and (acc/person) > max_avg_acc3:
                max_avg_acc3 = acc/person

            if writer is not None:
                writer.write(img)

            cv2.imshow("Human Detection from Camera", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

        # Prepare max lines for plotting
        for _ in framex3:
            max3.append(max_count3)
            max_avg_acc3_list.append(max_avg_acc3)

        info1.config(text="Status: Detection Completed")

        # IMPORTANT → Enable buttons AFTER detection finishes
        btn_enum.config(state="normal")
        btn_acc.config(state="normal")
        btn_report.config(state="normal")


    # ---------- Camera Thread ----------
    def start_camera():
        global max_count3, framex3, county3, max3
        global avg_acc3_list, max_avg_acc3_list, max_acc3, max_avg_acc3

        # -------- HARD RESET --------
        max_count3 = 0
        framex3 = []
        county3 = []
        max3 = []
        avg_acc3_list = []
        max_avg_acc3_list = []
        max_acc3 = 0
        max_avg_acc3 = 0

        # Disable buttons until detection completes
        btn_enum.config(state="disabled")
        btn_acc.config(state="disabled")
        btn_report.config(state="disabled")

        info1.config(text="Status: Opening Camera...")

        # Start detection thread
        threading.Thread(target=detectByCamera, daemon=True).start()


    # ---------- Plot & Report Functions ----------
    def cam_enumeration_plot():
        if not framex3:
            mbox.showwarning("No Data", "Run camera detection first.", parent=windowc)
            return
        plt.figure(facecolor='white')
        ax = plt.axes()
        ax.set_facecolor("#e0e0e0")
        plt.plot(framex3, county3, label="Human Count", marker='o', color="#2196f3")
        plt.plot(framex3, max3, linestyle='dashed', label="Max Count", color="#f44336")
        plt.xlabel("Time (sec)")
        plt.ylabel("Human Count")
        plt.legend()
        plt.title("Enumeration Plot")
        plt.show()

    def cam_accuracy_plot():
        if not framex3:
            mbox.showwarning("No Data", "Run camera detection first.", parent=windowc)
            return
        plt.figure(facecolor='white')
        ax = plt.axes()
        ax.set_facecolor("#e0e0e0")
        plt.plot(framex3, avg_acc3_list, label="Avg Accuracy", marker='o', color="#4caf50")
        plt.plot(framex3, max_avg_acc3_list, linestyle='dashed', label="Max Avg Accuracy", color="#f44336")
        plt.xlabel("Time (sec)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Avg Accuracy Plot")
        plt.show()

    # def cam_gen_report():
    #     pdf = FPDF()
    #     pdf.add_page()
    #     pdf.set_font("Arial", "", 20)
    #     pdf.set_text_color(128,0,0)
    #     pdf.image('Images/Crowd_Report.png', x=0, y=0, w=210, h=297)

    #     pdf.text(125, 150, str(max_count3))
    #     pdf.text(105, 163, str(max_acc3))
    #     pdf.text(125, 175, str(max_avg_acc3))

    #     if max_count3 > 25:
    #         pdf.text(26,220,"Max. Human Detected above MAX LIMIT.")
    #         pdf.text(70,235,"Region is Crowded.")
    #     else:
    #         pdf.text(26,220,"Max. Human Detected within MAX LIMIT.")
    #         pdf.text(65,235,"Region is not Crowded.")

    #     pdf.output('Crowd_Report.pdf')
    #     mbox.showinfo("Status", "Report Generated Successfully.", parent=windowc)
    
    def cam_gen_report():
        if not framex3:
            mbox.showwarning("No Data", "Run camera detection first.", parent=windowc)
            return

        pdf = FPDF()
        pdf.add_page()

        # ===== Title =====
        pdf.set_font("Arial", "B", 20)
        pdf.cell(0, 12, "Human Detection Analysis Report", ln=True, align="C")

        pdf.ln(5)

        # ===== Input Information =====
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Input Information", ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, "Source Type: Live Camera", ln=True)
        pdf.cell(0, 8, "Detection Threshold: 0.7", ln=True)

        pdf.ln(5)

        # ===== Compute Statistics =====
        total_frames = len(framex3)
        avg_count = sum(county3) / total_frames if total_frames else 0
        min_count = min(county3) if county3 else 0
        variation = max(county3) - min_count if county3 else 0
        confidence = (
            "High" if max_avg_acc3 > 0.85 else
            "Moderate" if max_avg_acc3 > 0.70 else
            "Low"
        )

        if max_count3 > 25:
            crowd_status = "High Density Crowd"
        elif max_count3 > 10:
            crowd_status = "Moderate Crowd"
        else:
            crowd_status = "Low Crowd Presence"

        # ===== Detection Summary Table =====
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Detection Summary", ln=True)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(95, 10, "Metric", border=1)
        pdf.cell(95, 10, "Value", border=1, ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.cell(95, 10, "Maximum Humans Detected", border=1)
        pdf.cell(95, 10, str(max_count3), border=1, ln=True)

        pdf.cell(95, 10, "Average Humans Detected", border=1)
        pdf.cell(95, 10, str(round(avg_count, 2)), border=1, ln=True)

        pdf.cell(95, 10, "Minimum Humans Detected", border=1)
        pdf.cell(95, 10, str(min_count), border=1, ln=True)

        pdf.cell(95, 10, "Maximum Detection Accuracy", border=1)
        pdf.cell(95, 10, str(round(max_acc3, 2)), border=1, ln=True)

        pdf.cell(95, 10, "Average Detection Accuracy", border=1)
        pdf.cell(95, 10, str(round(max_avg_acc3, 2)), border=1, ln=True)

        pdf.cell(95, 10, "Total Frames Analysed", border=1)
        pdf.cell(95, 10, str(total_frames), border=1, ln=True)

        pdf.ln(5)

        # ===== Statistical Analysis =====
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Statistical Analysis", ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Crowd Density Level: {crowd_status}", ln=True)
        pdf.cell(0, 8, f"Detection Confidence Level: {confidence}", ln=True)
        pdf.cell(0, 8, f"Crowd Variation Range: {variation}", ln=True)

        # ===== Save + Open =====
        report_path = os.path.abspath("Crowd_Report.pdf")

        try:
            if os.path.exists(report_path):
                os.remove(report_path)
        except:
            pass

        pdf.output(report_path)
        mbox.showinfo("Status", "Report Generated Successfully.", parent=windowc)
        os.startfile(report_path)

    # ---------- Buttons (always visible) ----------
    btn_start = tk.Button(frame_action, text="Open Camera", command=start_camera, font=("Segoe UI", 12), bg="#4caf50", fg="white")
    btn_start.pack(side="left", padx=10, pady=10)

    btn_enum = tk.Button(frame_action, text="Enumeration Plot", command=cam_enumeration_plot, state="disabled", font=("Segoe UI", 12), bg="#ff9800", fg="white")
    btn_enum.pack(side="left", padx=10, pady=10)

    btn_acc = tk.Button(frame_action, text="Avg Accuracy Plot", command=cam_accuracy_plot, state="disabled", font=("Segoe UI", 12), bg="#ff9800", fg="white")
    btn_acc.pack(side="left", padx=10, pady=10)

    btn_report = tk.Button(frame_action, text="Generate Report", command=cam_gen_report, state="disabled", font=("Segoe UI", 12), bg="#ff9800", fg="white")
    btn_report.pack(side="left", padx=10, pady=10)

    # ---------- Exit ----------
    def exit_winc():
        if mbox.askokcancel("Exit", "Do you want to exit?", parent=windowc):
            try:
                cv2.destroyAllWindows()
            except:
                pass
            windowc.destroy()

    windowc.protocol("WM_DELETE_WINDOW", exit_winc)
    windowc.mainloop()

    
# ---------------------------- options section ------------------------------------------------------------

# Background color
window1.configure(bg="#EEF1F5")

# ---------- TITLE ----------
title_options = tk.Label(
    window1,
    text="SYSTEM OPTIONS",
    font=("Segoe UI", 34, "bold"),
    fg="#0A1F44",
    bg="#EEF1F5"
)
title_options.place(relx=0.5, rely=0.08, anchor="center")


# ---------- BUTTON STYLE ----------
btn_style = {
    "font": ("Segoe UI", 18, "bold"),
    "width": 26,
    "height": 2,
    "bd": 0,
    "cursor": "hand2"
}


# ---------- IMAGE ICONS ----------
imgi = ImageTk.PhotoImage(Image.open("Images/image1.PNG").resize((120, 120)))
imgv = ImageTk.PhotoImage(Image.open("Images/image2.PNG").resize((120, 120)))
imgc = ImageTk.PhotoImage(Image.open("Images/image3.PNG").resize((120, 120)))

icon1 = tk.Label(window1, image=imgi, bg="#EEF1F5")
icon1.place(relx=0.32, rely=0.30, anchor="center")

icon2 = tk.Label(window1, image=imgv, bg="#EEF1F5")
icon2.place(relx=0.32, rely=0.48, anchor="center")

icon3 = tk.Label(window1, image=imgc, bg="#EEF1F5")
icon3.place(relx=0.32, rely=0.66, anchor="center")


# ---------- BUTTONS ----------
btn_image = tk.Button(
    window1,
    text="DETECT FROM IMAGE",
    command=image_option,
    bg="#1F3A5F",
    fg="white",
    activebackground="#162B47",
    **btn_style
)
btn_image.place(relx=0.60, rely=0.30, anchor="center")


btn_video = tk.Button(
    window1,
    text="DETECT FROM VIDEO",
    command=video_option,
    bg="#2E5C8A",
    fg="white",
    activebackground="#24496E",
    **btn_style
)
btn_video.place(relx=0.60, rely=0.48, anchor="center")


btn_camera = tk.Button(
    window1,
    text="DETECT FROM CAMERA",
    command=camera_option,
    bg="#3C6E71",
    fg="white",
    activebackground="#2F5558",
    **btn_style
)
btn_camera.place(relx=0.60, rely=0.66, anchor="center")

def exit_win1():
    if mbox.askokcancel("Exit", "Do you want to exit?"):
        window1.destroy()
        
# ---------- EXIT BUTTON ----------
exit_btn = tk.Button(
    window1,
    text="EXIT SYSTEM",
    command=exit_win1,
    font=("Segoe UI", 16, "bold"),
    bg="#6B0F1A",
    fg="white",
    activebackground="#4A0A12",
    width=18,
    height=1,
    bd=0,
    cursor="hand2"
)
exit_btn.place(relx=0.5, rely=0.88, anchor="center")

window1.protocol("WM_DELETE_WINDOW", exit_win1)
window1.mainloop()





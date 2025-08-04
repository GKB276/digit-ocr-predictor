from tensorflow.keras.models import load_model
import numpy as np
import cv2, sys, os
from tkinter import *
from tkinter import ttk, filedialog
import mysql.connector

if hasattr(sys, 'frozen'):
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    

ans = []
file_paths = []
confidences = []
cropped_img = []
final_ans = []
final_confidence = []

def error_check():
    if error_var.get() == "":
        pass
    else:
        process_var.set("True")
    
    
def browse_file():
    return filedialog.askdirectory(title="Select root folder")

def get_image_files(root_folder):
    valid_extensions = (".png", ".jpg", ".jpeg")
    image_files = []
    for dirpath, dirs, filenames in os.walk(root_folder):
        for file in filenames:
            full_path = os.path.join(dirpath, file)
            if full_path.lower().endswith(valid_extensions):
                image_files.append(full_path)
    return image_files

def validate_mysql ():
    conn = connect_mysql(host_var.get(), username_var.get(), pass_var.get())
    cursor = conn.cursor()
    check_db(cursor, db_var.get())
    cursor.close()
    conn.close()

    conn = connect_db(host_var.get(), username_var.get(), pass_var.get(), db_var.get())
    cursor = conn.cursor()
    check_table(cursor, table="Prediction")
    
    return conn

def segmentation(image_files):
    final_ans.clear()
    final_confidence.clear()
    
    conn = validate_mysql()

    for index,i in enumerate(image_files):
        error_check()
        if process_var.get() == "True":
            break
        
        cropped_img.clear()
        img = cv2.imread(i)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(img_gray, 95, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours_line = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
        
        progress_bar['maximum'] = len(image_files)
        progress_bar['value'] = index + 1
        prog_var.set(f"Progress: {index + 1}/{len(image_files)}")
        root.update_idletasks()
        for ctr in sorted_contours_line:
            x, y, w, h = cv2.boundingRect(ctr)
            if h < 10 or w < 4:
                continue

            pad = 7
            x1, y1 = max(x - pad, 0), max(y - pad, 0)
            x2, y2 = x + w + pad, y + h + pad
            roi = img[y1:y2, x1:x2]
            cropped_img.append(roi)

        if not cropped_img:
            final_ans.append("*")
            final_confidence.append(0.0)
            continue
            
        else:
            result, conf_avg = predict_digits(cropped_img)
            final_ans.append(result)
            final_confidence.append(conf_avg)

    start(conn,file_paths, final_ans, final_confidence)

model = load_model('MNIST_keras_CNN.h5', compile=False)

def predict_digits(images):
    error_var.set("")
    prog_var.set("")
    ans.clear()
    confidences.clear()

    if not images:
        error_var.set("No images to predict.")
        error_check()
        return "", 0.0

    for digit in images:
        error_check()
        if process_var.get() == "True":
            break
        
        try:
            digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
            digit = cv2.resize(digit, (28, 28))
            digit = 255 - digit
            digit = digit / 255.0
            digit = digit.reshape(1, 28, 28, 1)

            predicted = model.predict(digit)
            confidence_scores = predicted[0] * 100
            predicted_digit = np.argmax(predicted)
            
            confidence = confidence_scores[predicted_digit]
            val = round(float(confidence), 2)
            
            if val < 40:
                predicted_digit = "*"
                ans.append(predicted_digit)
                val = 0
            else:    
                ans.append(int(predicted_digit))
            confidences.append(val)


        except Exception as e:
            error_var.set(f"Error processing image: {e}")
            error_check()

    result = ''.join(str(x) for x in ans)
    conf_avg = round((sum(confidences) / len(confidences)),2) 
    prog_var.set("Completed!")

    return result, conf_avg

def access_info ():
    with open("access_info.txt", "r") as file:
        content = file.read()
    
    (host,user,passw,db) = content.splitlines()
    host_var.set(host)
    username_var.set(user)
    pass_var.set(passw)
    db_var.set(db)



def start_pred():
    folder = browse_file()
    access_info()
    if folder:
        selected_images = get_image_files(folder)
        file_paths.clear()
        file_paths.extend(selected_images)
        if selected_images:
            path_var.set(f"Selected folder: {folder}")
        else:
            path_var.set("No image files found.")
    else:
        path_var.set("No folder selected.")

def connect_mysql(host, user, password):
    try:
        return mysql.connector.connect(host=host, user=user, password=password)
    except Exception as e:
        error_var.set(e)
        error_check()

def check_db(cursor, database):
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")

def check_table(cursor, table):
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            Source VARCHAR(255),
            Predicted_Digit VARCHAR(20),
            Confidence FLOAT
        )
    """)

def connect_db(host, user, password, database):
    try:
        return mysql.connector.connect(host=host, user=user, password=password, database=database)
    except Exception as e:
        error_var.set(e)
        error_check()

def start(conn,files, answers, confidences):
    
    cursor = conn.cursor()

    for i in range(len(files)):
        error_check()
        if process_var.get() == "True":
            break
        sql = "INSERT INTO Prediction (Source, Predicted_Digit, Confidence) VALUES (%s, %s, %s)"
        val = (files[i], answers[i], confidences[i])
        cursor.execute(sql, val)

    conn.commit()
    cursor.close()
    conn.close()

root = Tk()
root.title("Digit Recognition")
root.geometry("600x400")

frm = ttk.Frame(root, padding=(10, 10))
frm.grid(column=0, row=0, sticky="nsew")
root.columnconfigure(0, weight=1)
frm.columnconfigure(0, weight=1)
frm.columnconfigure(1, weight=1)

path_var = StringVar()
prog_var = StringVar()
host_var = StringVar()
username_var = StringVar()
pass_var = StringVar()
db_var = StringVar()
error_var = StringVar()
process_var = StringVar()
error_var.set("")
process_var.set("False")

ttk.Button(frm, text="Choose Folder", command=start_pred).grid(column=0, row=5, columnspan=2, pady=5)
ttk.Label(frm, textvariable=path_var).grid(column=0, row=6, columnspan=2, pady=2)
ttk.Button(frm, text="Predict & Save", command=lambda: segmentation(file_paths)).grid(column=0, row=7, columnspan=2, pady=5)
ttk.Label(frm, textvariable=prog_var).grid(column=0, row=8, columnspan=2, pady=2)

progress_bar = ttk.Progressbar(frm, orient=HORIZONTAL, length=300, mode='determinate')
progress_bar.grid(column=0, row=9, columnspan=2, pady=10)
ttk.Label(frm, textvariable=error_var).grid(column=0, row=10, columnspan=2, pady=2)

root.mainloop()

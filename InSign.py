# Arranged by Adya Muhammad Prawira

import os
import json
import csv
import cv2
import re
from PIL import Image, ImageTk
import pickle
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import mediapipe as mp
import pandas as pd
import numpy as np
import openpyxl
from collections import Counter
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
from sklearn.exceptions import DataConversionWarning

cap = cv2.VideoCapture(0)
cap.release()

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

count = 0
state = 'home_frame'
add_dataset_holistic_state = ''
update_dataset_retraining_holistic_state = ''
translate_state = ''
appearance_mode = 'system'
color_theme = 'green'
appearance_mode_values = ['system', 'dark', 'light']
color_theme_values = ['green', 'blue', 'dark-blue']
all_buttons = []
all_switches = []
all_progress_bar = []
check_translate = []
all_second_buttons = []
all_second_switches = []
training_buttons = None
data_class = 500
training_result_second_frame = None

pop_out_setting_state = False

notification_add_dataset_information_0 = 'Please enter a new word!'
notification_add_dataset_information_1 = 'The word already exists!'

notification_update_dataset_rewording_0 = 'Please complete the words input!'
notification_update_dataset_rewording_1 = 'Word not found in dataset!'
notification_update_dataset_rewording_2 = 'New word found in the dataset!'

notification_update_dataset_retraining_information_0 = 'Please enter a word!'
notification_update_dataset_retraining_information_1 = 'Word not found in dataset!'

notification_delete_dataset_0 = 'Please enter a word!'
notification_delete_dataset_1 = 'Word not found in dataset!'

notification_add_dataset_holistic_0 = '''notification:

Click Start Button!
'''
notification_add_dataset_holistic_1 = '''notification:

Perform pose variations to enrich the dataset!
'''
notification_add_dataset_holistic_2 = '''notification:

The dataset has been trained!
'''
notification_update_dataset_retraining_holistic_0 = '''notification:

Click Start Button!
'''
notification_update_dataset_retraining_holistic_1 = '''notification:

Perform pose variations to enrich the dataset!
'''
notification_update_dataset_retraining_holistic_2 = '''notification:

The dataset has been trained!
'''

app = ctk.CTk()
app.geometry('600x350')
app.geometry('+50+50')
app.resizable(width=False, height=False)
app.iconbitmap('D:/Project/InSign/assets/InSign.ico')

app.title('In Sign (Home)')

ctk.set_appearance_mode(appearance_mode)
ctk.set_default_color_theme(color_theme)

# =======================================================================================================================================================    

def check_dataset():
    header = ['class']
    x_right_array = []
    y_right_array = []
    x_left_array = []
    y_left_array = []
    for i in range(21):
        x_right_array.append(f'rx{i}')
        y_right_array.append(f'ry{i}')
        x_left_array.append(f'lx{i}')
        y_left_array.append(f'ly{i}')
    header.extend(x_right_array + y_right_array + x_left_array + y_left_array)
    if os.path.isfile('dataset.csv'):
        csv_file = open('dataset.csv', 'r', newline='')
        csv_reader = csv.reader(csv_file)        
        first_row = next(csv_reader, None)    
        if not first_row == header:
            csv_file = open('dataset.csv', 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(header)
    else:
        csv_file = open('dataset.csv', 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

check_dataset()

def training_dataset():
    df = pd.read_csv('dataset.csv')
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    pipelines = {
        'rf': make_pipeline(StandardScaler(), RandomForestClassifier())
    }
    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model
    fit_models['rf'].predict(X_test)
    for algo, model in fit_models.items():
        y_predict = model.predict(X_test)
    with open('data_training.pkl', 'wb') as file:
        pickle.dump(fit_models['rf'], file)

def check_data_training():    
    if not os.path.isfile('data_training.pkl'):
        csv_file = open('dataset.csv', 'r', newline='')
        csv_reader = csv.reader(csv_file)
        row_count = sum(1 for row in csv_reader)
        if row_count > data_class:
            training_dataset()
        else:
            messagebox.showwarning("Warning", "You must add datasets to use some of the features of this application!")

check_data_training()

def check_valid_date_time_format(input_string, date_format='%Y-%m-%d_%H-%M-%S'):
    if input_string == '0000-00-00_00-00-00':
        return True
    try:
        datetime.strptime(input_string, date_format)
        return True
    except ValueError:
        return False
    
def check_history():
    if os.path.isfile('history.xlsx'):
        excel_data = pd.read_excel('history.xlsx', sheet_name=None)
        for sheet_name, sheet_data in excel_data.items():
            if not check_valid_date_time_format(sheet_name):
                os.remove('history.xlsx')
                break
            if not sheet_data.columns.tolist() == ['Translation', 'Date', 'Time']:
                os.remove('history.xlsx')
                break
check_history()

# =======================================================================================================================================================    

def add_dataset_holistic_camera():
    global state, add_dataset_holistic_state, cap, count, mp_drawing, mp_holistic, holistic, add_dataset_model
    if state == 'add_dataset_holistic_frame':
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = holistic.process(frame)        
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if bone_add_dataset_holistic_switch.get() == 'on':
            if left_landmark_switch.get() == 'on':            
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
                )
            if right_landmark_switch.get() == 'on':
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
                )
            if pose_landmark_switch.get() == 'on':
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0,125,255), thickness=2, circle_radius=2)
                )
            if face_landmark_switch.get() == 'on':
                mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255,255,0), thickness=1, circle_radius=1)
                )                
            if left_rectangle_switch.get() == 'on':
                if results.left_hand_landmarks:            
                    left_hand_point = results.left_hand_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in left_hand_point]
                    y_coordinates = [landmark.y for landmark in left_hand_point]
                    x_min = int(min(x_coordinates) * frame.shape[1]) - 30
                    y_min = int(min(y_coordinates) * frame.shape[0]) - 30
                    x_max = int(max(x_coordinates) * frame.shape[1]) + 30
                    y_max = int(max(y_coordinates) * frame.shape[0]) + 30
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)                                
            if right_rectangle_switch.get() == 'on':
                if results.right_hand_landmarks:            
                    right_hand_point = results.right_hand_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in right_hand_point]
                    y_coordinates = [landmark.y for landmark in right_hand_point]
                    x_min = int(min(x_coordinates) * frame.shape[1]) - 30
                    y_min = int(min(y_coordinates) * frame.shape[0]) - 30
                    x_max = int(max(x_coordinates) * frame.shape[1]) + 30
                    y_max = int(max(y_coordinates) * frame.shape[0]) + 30
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            if face_rectangle_switch.get() == 'on':
                if results.face_landmarks:
                    face_point = results.face_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in face_point]
                    y_coordinates = [landmark.y for landmark in face_point]
                    x_min = int(min(x_coordinates) * frame.shape[1]) - 30
                    y_min = int(min(y_coordinates) * frame.shape[0]) - 30
                    x_max = int(max(x_coordinates) * frame.shape[1]) + 30
                    y_max = int(max(y_coordinates) * frame.shape[0]) + 30
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (170,100,255), 2)
        if add_dataset_holistic_state == 'start_button':
            if results.right_hand_landmarks or results.left_hand_landmarks:
                right_x_initial = [0] * 21
                right_y_initial = [0] * 21
                left_x_initial = [0] * 21
                left_y_initial = [0] * 21
                right_x_min = 0
                right_y_min = 0
                left_x_min = 0
                left_y_min = 0
                if results.right_hand_landmarks:
                    right_hand_point = results.right_hand_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in right_hand_point]
                    y_coordinates = [landmark.y for landmark in right_hand_point]
                    right_x_min = min(x_coordinates) * frame.shape[1]
                    right_y_min = min(y_coordinates) * frame.shape[0]
                    x_coordinates = np.array(x_coordinates) * frame.shape[1]
                    y_coordinates = np.array(y_coordinates) * frame.shape[0]
                    right_x_initial = list(x_coordinates - right_x_min)
                    right_y_initial = list(y_coordinates - right_y_min)
                if results.left_hand_landmarks:
                    left_hand_point = results.left_hand_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in left_hand_point]
                    y_coordinates = [landmark.y for landmark in left_hand_point]
                    left_x_min = min(x_coordinates) * frame.shape[1]
                    left_y_min = min(y_coordinates) * frame.shape[0]
                    x_coordinates = np.array(x_coordinates) * frame.shape[1]
                    y_coordinates = np.array(y_coordinates) * frame.shape[0]
                    left_x_initial = list(x_coordinates - left_x_min)
                    left_y_initial = list(x_coordinates - left_y_min)
                new_word = new_word_add_dataset_information_entry.get().title()                                    
                total_row = [new_word] + right_x_initial + right_y_initial + left_x_initial + left_y_initial
                csv_file = open('dataset.csv', 'a', newline='')
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(total_row)
                count += 1
                percentage = (count / data_class)
                add_dataset_holistic_progress_bar.set(percentage)
                add_dataset_holistic_progress_label.configure(text=f'{percentage*100:.2f}%')                
                if count >= data_class:
                    training_add_dataset_holistic_data()       
        elif add_dataset_holistic_state == 'test_button':
            cv2.rectangle(frame, (0, 0), (300, 80), (255, 255, 255), -1)
            cv2.putText(frame, 'Class:', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Prob:', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            if results.right_hand_landmarks or results.left_hand_landmarks:            
                right_x_initial = [0] * 21
                right_y_initial = [0] * 21
                left_x_initial = [0] * 21
                left_y_initial = [0] * 21
                right_x_min = 0
                right_y_min = 0
                left_x_min = 0
                left_y_min = 0
                if results.right_hand_landmarks:
                    right_hand_point = results.right_hand_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in right_hand_point]
                    y_coordinates = [landmark.y for landmark in right_hand_point]
                    right_x_min = min(x_coordinates) * frame.shape[1]
                    right_y_min = min(y_coordinates) * frame.shape[0]
                    x_coordinates = np.array(x_coordinates) * frame.shape[1]
                    y_coordinates = np.array(y_coordinates) * frame.shape[0]
                    right_x_initial = list(x_coordinates - right_x_min)
                    right_y_initial = list(y_coordinates - right_y_min)
                if results.left_hand_landmarks:
                    left_hand_point = results.left_hand_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in left_hand_point]
                    y_coordinates = [landmark.y for landmark in left_hand_point]
                    left_x_min = min(x_coordinates) * frame.shape[1]
                    left_y_min = min(y_coordinates) * frame.shape[0]
                    x_coordinates = np.array(x_coordinates) * frame.shape[1]
                    y_coordinates = np.array(y_coordinates) * frame.shape[0]
                    left_x_initial = list(x_coordinates - left_x_min)
                    left_y_initial = list(x_coordinates - left_y_min)                
                total_row = right_x_initial + right_y_initial + left_x_initial + left_y_initial                
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=DataConversionWarning)
                X_predict = pd.DataFrame([total_row])
                body_language_class = add_dataset_model.predict(X_predict)[0]
                body_language_proba = add_dataset_model.predict_proba(X_predict)[0]                
                cv2.putText(frame, body_language_class, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f'{round((body_language_proba[np.argmax(body_language_proba)] * 100), 2)}%', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'Unknown', (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, '-', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)            
        camera_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        camera_image = camera_image.resize((815, 585), Image.LANCZOS)
        camera_imgtk = ImageTk.PhotoImage(image=camera_image)
        add_dataset_holistic_camera_label.imgtk = camera_imgtk
        add_dataset_holistic_camera_label.configure(image=camera_imgtk)
        add_dataset_holistic_camera_label.after(10, add_dataset_holistic_camera)

def update_dataset_retraining_holistic_camera():
    global state, update_dataset_retraining_holistic_state, cap, count, mp_drawing, mp_holistic, holistic, update_dataset_retraining_model
    if state == 'update_dataset_retraining_holistic_frame':        
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = holistic.process(frame)        
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if bone_update_dataset_retraining_holistic_switch.get() == 'on':
            if left_landmark_switch.get() == 'on':            
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
                )
            if right_landmark_switch.get() == 'on':
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
                )
            if pose_landmark_switch.get() == 'on':
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0,125,255), thickness=2, circle_radius=2)
                )
            if face_landmark_switch.get() == 'on':
                mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255,255,0), thickness=1, circle_radius=1)
                )                
            if left_rectangle_switch.get() == 'on':
                if results.left_hand_landmarks:            
                    left_hand_point = results.left_hand_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in left_hand_point]
                    y_coordinates = [landmark.y for landmark in left_hand_point]
                    x_min = int(min(x_coordinates) * frame.shape[1]) - 30
                    y_min = int(min(y_coordinates) * frame.shape[0]) - 30
                    x_max = int(max(x_coordinates) * frame.shape[1]) + 30
                    y_max = int(max(y_coordinates) * frame.shape[0]) + 30
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)                                
            if right_rectangle_switch.get() == 'on':
                if results.right_hand_landmarks:            
                    right_hand_point = results.right_hand_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in right_hand_point]
                    y_coordinates = [landmark.y for landmark in right_hand_point]
                    x_min = int(min(x_coordinates) * frame.shape[1]) - 30
                    y_min = int(min(y_coordinates) * frame.shape[0]) - 30
                    x_max = int(max(x_coordinates) * frame.shape[1]) + 30
                    y_max = int(max(y_coordinates) * frame.shape[0]) + 30
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            if face_rectangle_switch.get() == 'on':
                if results.face_landmarks:
                    face_point = results.face_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in face_point]
                    y_coordinates = [landmark.y for landmark in face_point]
                    x_min = int(min(x_coordinates) * frame.shape[1]) - 30
                    y_min = int(min(y_coordinates) * frame.shape[0]) - 30
                    x_max = int(max(x_coordinates) * frame.shape[1]) + 30
                    y_max = int(max(y_coordinates) * frame.shape[0]) + 30
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (170,100,255), 2)
        if update_dataset_retraining_holistic_state == 'start_button':
            if results.right_hand_landmarks or results.left_hand_landmarks:
                right_x_initial = [0] * 21
                right_y_initial = [0] * 21
                left_x_initial = [0] * 21
                left_y_initial = [0] * 21
                right_x_min = 0
                right_y_min = 0
                left_x_min = 0
                left_y_min = 0
                if results.right_hand_landmarks:
                    right_hand_point = results.right_hand_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in right_hand_point]
                    y_coordinates = [landmark.y for landmark in right_hand_point]
                    right_x_min = min(x_coordinates) * frame.shape[1]
                    right_y_min = min(y_coordinates) * frame.shape[0]
                    x_coordinates = np.array(x_coordinates) * frame.shape[1]
                    y_coordinates = np.array(y_coordinates) * frame.shape[0]
                    right_x_initial = list(x_coordinates - right_x_min)
                    right_y_initial = list(y_coordinates - right_y_min)
                if results.left_hand_landmarks:
                    left_hand_point = results.left_hand_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in left_hand_point]
                    y_coordinates = [landmark.y for landmark in left_hand_point]
                    left_x_min = min(x_coordinates) * frame.shape[1]
                    left_y_min = min(y_coordinates) * frame.shape[0]
                    x_coordinates = np.array(x_coordinates) * frame.shape[1]
                    y_coordinates = np.array(y_coordinates) * frame.shape[0]
                    left_x_initial = list(x_coordinates - left_x_min)
                    left_y_initial = list(x_coordinates - left_y_min)
                new_word = new_word_add_dataset_information_entry.get().title()                                    
                total_row = [new_word] + right_x_initial + right_y_initial + left_x_initial + left_y_initial
                csv_file = open('dataset.csv', 'a', newline='')
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(total_row)
                count += 1
                percentage = (count / data_class)
                add_dataset_holistic_progress_bar.set(percentage)
                add_dataset_holistic_progress_label.configure(text=f'{percentage*100:.2f}%')                
                if count >= data_class:
                    training_add_dataset_holistic_data()  
        elif update_dataset_retraining_holistic_state == 'test_button':
            cv2.rectangle(frame, (0, 0), (300, 80), (255, 255, 255), -1)
            cv2.putText(frame, 'Class:', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Prob:', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            if results.right_hand_landmarks or results.left_hand_landmarks:            
                right_x_initial = [0] * 21
                right_y_initial = [0] * 21
                left_x_initial = [0] * 21
                left_y_initial = [0] * 21
                right_x_min = 0
                right_y_min = 0
                left_x_min = 0
                left_y_min = 0
                if results.right_hand_landmarks:
                    right_hand_point = results.right_hand_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in right_hand_point]
                    y_coordinates = [landmark.y for landmark in right_hand_point]
                    right_x_min = min(x_coordinates) * frame.shape[1]
                    right_y_min = min(y_coordinates) * frame.shape[0]
                    x_coordinates = np.array(x_coordinates) * frame.shape[1]
                    y_coordinates = np.array(y_coordinates) * frame.shape[0]
                    right_x_initial = list(x_coordinates - right_x_min)
                    right_y_initial = list(y_coordinates - right_y_min)
                if results.left_hand_landmarks:
                    left_hand_point = results.left_hand_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in left_hand_point]
                    y_coordinates = [landmark.y for landmark in left_hand_point]
                    left_x_min = min(x_coordinates) * frame.shape[1]
                    left_y_min = min(y_coordinates) * frame.shape[0]
                    x_coordinates = np.array(x_coordinates) * frame.shape[1]
                    y_coordinates = np.array(y_coordinates) * frame.shape[0]
                    left_x_initial = list(x_coordinates - left_x_min)
                    left_y_initial = list(x_coordinates - left_y_min)                
                total_row = right_x_initial + right_y_initial + left_x_initial + left_y_initial     
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=DataConversionWarning)
                X_predict = pd.DataFrame([total_row])
                body_language_class = update_dataset_retraining_model.predict(X_predict)[0]
                body_language_proba = update_dataset_retraining_model.predict_proba(X_predict)[0]                
                cv2.putText(frame, body_language_class, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f'{round((body_language_proba[np.argmax(body_language_proba)] * 100), 2)}%', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'Unknown', (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, '-', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)             
        camera_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        camera_image = camera_image.resize((815, 585), Image.LANCZOS)
        camera_imgtk = ImageTk.PhotoImage(image=camera_image)
        update_dataset_retraining_holistic_camera_label.imgtk = camera_imgtk
        update_dataset_retraining_holistic_camera_label.configure(image=camera_imgtk)
        update_dataset_retraining_holistic_camera_label.after(10, update_dataset_retraining_holistic_camera)

def translate_camera():
    global state, translate_state, cap, count, mp_drawing, mp_holistic, holistic, translate_model, check_translate
    if state == 'translate_frame':        
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = holistic.process(frame)        
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if bone_translate_switch.get() == 'on':
            if left_landmark_switch.get() == 'on':            
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
                )
            if right_landmark_switch.get() == 'on':
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
                )
            if pose_landmark_switch.get() == 'on':
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0,125,255), thickness=2, circle_radius=2)
                )
            if face_landmark_switch.get() == 'on':
                mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255,255,0), thickness=1, circle_radius=1)
                )                
            if left_rectangle_switch.get() == 'on':
                if results.left_hand_landmarks:            
                    left_hand_point = results.left_hand_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in left_hand_point]
                    y_coordinates = [landmark.y for landmark in left_hand_point]
                    x_min = int(min(x_coordinates) * frame.shape[1]) - 30
                    y_min = int(min(y_coordinates) * frame.shape[0]) - 30
                    x_max = int(max(x_coordinates) * frame.shape[1]) + 30
                    y_max = int(max(y_coordinates) * frame.shape[0]) + 30
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)                                
            if right_rectangle_switch.get() == 'on':
                if results.right_hand_landmarks:            
                    right_hand_point = results.right_hand_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in right_hand_point]
                    y_coordinates = [landmark.y for landmark in right_hand_point]
                    x_min = int(min(x_coordinates) * frame.shape[1]) - 30
                    y_min = int(min(y_coordinates) * frame.shape[0]) - 30
                    x_max = int(max(x_coordinates) * frame.shape[1]) + 30
                    y_max = int(max(y_coordinates) * frame.shape[0]) + 30
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            if face_rectangle_switch.get() == 'on':
                if results.face_landmarks:
                    face_point = results.face_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in face_point]
                    y_coordinates = [landmark.y for landmark in face_point]
                    x_min = int(min(x_coordinates) * frame.shape[1]) - 30
                    y_min = int(min(y_coordinates) * frame.shape[0]) - 30
                    x_max = int(max(x_coordinates) * frame.shape[1]) + 30
                    y_max = int(max(y_coordinates) * frame.shape[0]) + 30
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (170,100,255), 2)
        if translate_state == 'start_button':
            cv2.rectangle(frame, (0, 0), (300, 80), (255, 255, 255), -1)
            cv2.putText(frame, 'Class:', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Prob:', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            if results.right_hand_landmarks or results.left_hand_landmarks:            
                right_x_initial = [0] * 21
                right_y_initial = [0] * 21
                left_x_initial = [0] * 21
                left_y_initial = [0] * 21
                right_x_min = 0
                right_y_min = 0
                left_x_min = 0
                left_y_min = 0
                if results.right_hand_landmarks:
                    right_hand_point = results.right_hand_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in right_hand_point]
                    y_coordinates = [landmark.y for landmark in right_hand_point]
                    right_x_min = min(x_coordinates) * frame.shape[1]
                    right_y_min = min(y_coordinates) * frame.shape[0]
                    x_coordinates = np.array(x_coordinates) * frame.shape[1]
                    y_coordinates = np.array(y_coordinates) * frame.shape[0]
                    right_x_initial = list(x_coordinates - right_x_min)
                    right_y_initial = list(y_coordinates - right_y_min)
                if results.left_hand_landmarks:
                    left_hand_point = results.left_hand_landmarks.landmark
                    x_coordinates = [landmark.x for landmark in left_hand_point]
                    y_coordinates = [landmark.y for landmark in left_hand_point]
                    left_x_min = min(x_coordinates) * frame.shape[1]
                    left_y_min = min(y_coordinates) * frame.shape[0]
                    x_coordinates = np.array(x_coordinates) * frame.shape[1]
                    y_coordinates = np.array(y_coordinates) * frame.shape[0]
                    left_x_initial = list(x_coordinates - left_x_min)
                    left_y_initial = list(x_coordinates - left_y_min)                
                total_row = right_x_initial + right_y_initial + left_x_initial + left_y_initial     
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=DataConversionWarning)
                X_predict = pd.DataFrame([total_row])
                body_language_class = translate_model.predict(X_predict)[0]
                body_language_proba = translate_model.predict_proba(X_predict)[0]                
                cv2.putText(frame, f'{round((body_language_proba[np.argmax(body_language_proba)] * 100), 2)}%', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, body_language_class, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)                
                check_translate.append(body_language_class)
                if len(check_translate) == 15:
                    count_words = Counter(check_translate)
                    modus_words = [word for word, max_words in count_words.items() if max_words == max(count_words.values())]
                    text_output = translate_label.cget('text')
                    get_word = text_output.split(' ')
                    get_word = list(filter(None, text_output.split(' ')))
                    if len(get_word) == 1:
                        if not modus_words[0] == get_word[-1]:
                            text_output = text_output + modus_words[0] + ' '
                            translate_label.configure(text=text_output)
                            count = 0
                            translate_progress_bar.set(0)
                    elif len(get_word) > 1:
                        if not modus_words[0] == get_word[-1] and not modus_words[0] == get_word[-2] + ' ' + get_word[-1]:
                            text_output = text_output + modus_words[0] + ' '
                            translate_label.configure(text=text_output)
                            count = 0
                            translate_progress_bar.set(0)
                    else:
                        translate_label.configure(text=modus_words[0] + ' ')
                    check_translate = []                                                 
            else:
                cv2.putText(frame, 'Unknown', (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, '-', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)  
            if not translate_label.cget('text') == '':
                count += 1
                translate_progress_bar.set(count/100)
                if count == 100:
                    text_output = f'''{datetime.now().strftime("Date: %Y-%m-%d     Time: %H:%M:%S")}

{translate_label.cget('text')}

'''
                    output_translate_label = ctk.CTkLabel(master=list_translate_frame, text=text_output, font=('Roboto', 13), anchor='w', justify='left', wraplength=300)
                    output_translate_label.pack(fill='both')
                    translate_label.configure(text='')
                    translate_progress_bar.set(0)
                    count = 0
                    check_translate = []                                                 
        elif translate_state == 'stop_button':
            pass                
        camera_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        camera_image = camera_image.resize((815, 585), Image.LANCZOS)
        camera_imgtk = ImageTk.PhotoImage(image=camera_image)
        translate_camera_label.imgtk = camera_imgtk
        translate_camera_label.configure(image=camera_imgtk)
        translate_camera_label.after(10, translate_camera)

# =======================================================================================================================================================    

def safe_translations_to_history():
    if len(list_translate_frame.winfo_children()) > 0:        
        old_data = None
        if os.path.isfile('history.xlsx'):
            old_data = pd.read_excel('history.xlsx', sheet_name=None)
        sheet = ''
        data = {            
            'Translation': [],
            'Date': [],
            'Time': []
        }
        match = re.search(r'Date: (\d{4}-\d{2}-\d{2})\s+Time: (\d{2}:\d{2}:\d{2})\s+(.*)', list_translate_frame.winfo_children()[0].cget('text'))        
        if match:
            date = match.group(1)
            time = match.group(2)
            time = time.replace(':', '-')
            sheet = date + '_' + time
        else:
            sheet = '0000-00-00_00-00-00'         
        for translation in list_translate_frame.winfo_children():    
            match = re.search(r'Date: (\d{4}-\d{2}-\d{2})\s+Time: (\d{2}:\d{2}:\d{2})\s+(.*)', translation.cget('text'))        
            if match:
                date = match.group(1)
                time = match.group(2)
                translation = match.group(3)
                data['Translation'].append(translation)
                data['Date'].append(date)
                data['Time'].append(time)
            else:
                data['Translation'].append('Error')
                data['Date'].append('Error')
                data['Time'].append('Error')
        df = pd.DataFrame(data)    
        if old_data == None:
            with pd.ExcelWriter('history.xlsx', engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name=sheet, index=False)       
        else:
            old_data[sheet] = df
            with pd.ExcelWriter('history.xlsx', engine='xlsxwriter') as writer:
                for sheet_name, new_df in old_data.items():
                    new_df.to_excel(writer, sheet_name=sheet_name, index=False)            
            
# =======================================================================================================================================================    

def training_add_dataset_holistic_data():
    training_dataset()
    global count, add_dataset_holistic_state
    count = 0
    add_dataset_holistic_state = ''    
    start_add_dataset_holistic_button.grid_forget()
    add_dataset_holistic_progress_label.grid_forget()
    add_dataset_holistic_progress_bar.grid_forget()
    add_dataset_holistic_progress_bar.set(0)    
    home_add_dataset_holistic_button.grid(row=5, column=1, sticky='nsew', padx=5, pady=5)
    test_add_dataset_holistic_button.grid(row=6, column=3, sticky='nsew', padx=5, pady=5)
    back_add_dataset_holistic_button.grid(row=6, column=1, sticky='nsew', padx=5, pady=5)    
    add_dataset_holistic_notification_label.configure(text=notification_add_dataset_holistic_2)

def training_update_dataset_retraining_holistic_data():
    training_dataset()
    global count, update_dataset_retraining_holistic_state
    count = 0
    update_dataset_retraining_holistic_state = ''    
    start_update_dataset_retraining_holistic_button.grid_forget()
    update_dataset_retraining_holistic_progress_label.grid_forget()
    update_dataset_retraining_holistic_progress_bar.grid_forget()
    update_dataset_retraining_holistic_progress_bar.set(0)    
    home_update_dataset_retraining_holistic_button.grid(row=5, column=1, sticky='nsew', padx=5, pady=5)
    test_update_dataset_retraining_holistic_button.grid(row=6, column=3, sticky='nsew', padx=5, pady=5)
    back_update_dataset_retraining_holistic_button.grid(row=6, column=1, sticky='nsew', padx=5, pady=5)    
    update_dataset_retraining_holistic_notification_label.configure(text=notification_update_dataset_retraining_holistic_2)

# =======================================================================================================================================================    

def read_all_words():
    if os.path.isfile('data_training.pkl'):
        with open('data_training.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        class_names = loaded_model.classes_
        return class_names
    else:
        return []

def read_all_translations():
    if os.path.isfile('history.xlsx'):
        data = pd.read_excel('history.xlsx', sheet_name=None)
        get_data = []
        for sheet_name, sheet_data in data.items():
            temp_sheet = sheet_name.split('_')
            temp_sheet[1] = temp_sheet[1].replace('-', ':')
            sheet_id = temp_sheet[0] + ' ' + temp_sheet[1]
            get_data.append({
                'sheet': sheet_id,
                'values': sheet_data.values.tolist()
            })
        return get_data
    else:
        return []

# =======================================================================================================================================================    

def add_dataset_holistic_start_button_click():
    global count, add_dataset_holistic_state
    count = 0
    add_dataset_holistic_state = 'start_button'    
    home_add_dataset_holistic_button.grid_forget()
    test_add_dataset_holistic_button.grid_forget()
    start_add_dataset_holistic_button.grid_forget()
    back_add_dataset_holistic_button.grid_forget()
    add_dataset_holistic_progress_label.grid(row=6, column=2, sticky='nsew', padx=5, pady=5)
    add_dataset_holistic_progress_bar.grid(row=5, column=2, sticky='ew', padx=5, pady=5)
    add_dataset_holistic_progress_bar.set(0)
    add_dataset_holistic_notification_label.configure(text=notification_add_dataset_holistic_1)

def add_dataset_holistic_test_button_click():
    global add_dataset_holistic_state, add_dataset_model
    add_dataset_holistic_state = 'test_button'
    with open('data_training.pkl', 'rb') as read_file:
        add_dataset_model = pickle.load(read_file)

def update_dataset_retraining_holistic_start_button_click():
    global count, update_dataset_retraining_holistic_state
    count = 0
    update_dataset_retraining_holistic_state = 'start_button'    
    home_update_dataset_retraining_holistic_button.grid_forget()
    test_update_dataset_retraining_holistic_button.grid_forget()
    start_update_dataset_retraining_holistic_button.grid_forget()
    back_update_dataset_retraining_holistic_button.grid_forget()
    update_dataset_retraining_holistic_progress_label.grid(row=6, column=2, sticky='nsew', padx=5, pady=5)
    update_dataset_retraining_holistic_progress_bar.grid(row=5, column=2, sticky='ew', padx=5, pady=5)
    update_dataset_retraining_holistic_progress_bar.set(0)
    update_dataset_retraining_holistic_notification_label.configure(text=notification_update_dataset_retraining_holistic_1)
    get_word = choose_word_update_dataset_retraining_information_entry.get().title()                    
    df = pd.read_csv('dataset.csv')
    df = df[df['class'] != get_word]
    df.to_csv('dataset.csv', index=False)

def update_dataset_retraining_holistic_test_button_click():
    global update_dataset_retraining_holistic_state, update_dataset_retraining_model
    update_dataset_retraining_holistic_state = 'test_button'
    with open('data_training.pkl', 'rb') as read_file:
        update_dataset_retraining_model = pickle.load(read_file)

def delete_dataset_button_click():
    choose_word = choose_word_delete_dataset_entry.get()
    if choose_word == '':
        notification_delete_dataset_label.configure(text=notification_delete_dataset_0)
    else:
        words = read_all_words()
        if choose_word.title() in words:            
            get_word = choose_word.title()                   
            df = pd.read_csv('dataset.csv')
            df = df[df['class'] != get_word]
            df.to_csv('dataset.csv', index=False)
            csv_file = open('dataset.csv', 'r', newline='')
            csv_reader = csv.reader(csv_file)        
            row_count = sum(1 for row in csv_reader)
            if row_count == 1:
                os.remove('data_training.pkl')
                to_dataset_frame(delete_dataset_frame, dataset_frame)
                messagebox.showinfo('Notification', 'Your dataset and training data are empty!')
            else:
                training_dataset()
                for word in list_delete_dataset_frame.winfo_children():
                    word.destroy()
                words = read_all_words()
                for word in words:
                    word_label = ctk.CTkLabel(master=list_delete_dataset_frame, text=word, font=('Roboto', 15), anchor='w', justify='left')
                    word_label.pack(fill='both')
                notification_delete_dataset_label.configure(text=notification_delete_dataset_0)
                choose_word_delete_dataset_entry.delete(0, 'end')
                messagebox.showinfo('Notification', 'The word has been removed from your dataset and training data!')
        else:
            notification_delete_dataset_label.configure(text=notification_delete_dataset_1)

def update_dataset_rewording_button_click():
    choose_word = choose_word_update_dataset_rewording_entry.get()
    new_word = new_word_update_dataset_rewording_entry.get()
    if choose_word == '' or new_word == '':
        notification_update_dataset_rewording_label.configure(text=notification_update_dataset_rewording_0)
    else:
        words = read_all_words()
        check_rewording = 0
        if choose_word.title() in words:
            check_rewording += 1
        else:
            notification_update_dataset_rewording_label.configure(text=notification_update_dataset_rewording_1)
        if new_word.title() in words:
            notification_update_dataset_rewording_label.configure(text=notification_update_dataset_rewording_2)
        else:
            check_rewording += 1
        if check_rewording == 2:
            df = pd.read_csv('dataset.csv')
            df['class'] = df['class'].replace(choose_word.title(), new_word.title())
            df.to_csv('dataset.csv', index=False)
            training_dataset()
            for word in list_update_dataset_rewording_frame.winfo_children():
                word.destroy()
            words = read_all_words()
            for word in words:
                word_label = ctk.CTkLabel(master=list_update_dataset_rewording_frame, text=word, font=('Roboto', 15), anchor='w', justify='left')
                word_label.pack(fill='both')
            notification_update_dataset_rewording_label.configure(text=notification_update_dataset_rewording_0)
            messagebox.showinfo('Notification', 'The word in your dataset and training data has been changed!')

def translate_start_button_click():
    global count, translate_state, translate_model
    count = 0
    translate_state = 'start_button'  
    start_translate_button.grid_forget()
    back_translate_button.grid_forget()
    stop_translate_button.grid(row=5, column=4, sticky='nsew', padx=5, pady=5)
    bone_translate_switch.grid(row=4, column=4, sticky='w', padx=5, pady=5)
    translate_label.grid(row=4, column=2, columnspan=2, sticky='ew', padx=5, pady=5)
    translate_label.configure(text='')
    translate_progress_bar.grid(row=5, column=2, columnspan=2, sticky='ew', padx=5, pady=5)    
    translate_progress_bar.set(0)
    for translation in list_translate_frame.winfo_children():
        translation.destroy()
    with open('data_training.pkl', 'rb') as read_file:
        translate_model = pickle.load(read_file)

def translate_stop_button_click():
    global count, translate_state
    count = 0
    translate_state = ''    
    start_translate_button.grid(row=4, column=1, sticky='nsew', padx=5, pady=5)
    back_translate_button.grid(row=5, column=1, sticky='nsew', padx=5, pady=5)
    stop_translate_button.grid_forget()
    bone_translate_switch.grid(row=4, column=4, sticky='w', padx=5, pady=5)
    translate_label.grid_forget()
    translate_label.configure(text='')
    translate_progress_bar.grid_forget()
    translate_progress_bar.set(0)
    safe_translations_to_history()

def history_translation_label_click(event, param):
    for translation in list_translation_frame.winfo_children():
        translation.destroy()
    for translation in param:
        translation_text = f'''Date: {translation[1]}     Time: {translation[2]}

{translation[0]}
'''
        translation_label = ctk.CTkLabel(master=list_translation_frame, text=translation_text, font=('Roboto', 15), anchor='w', justify='left')
        translation_label.pack(fill='both')
    pass

def history_delete_button_click():
    if len(list_translation_frame.winfo_children()) > 0:
        match = re.search(r'Date: (\d{4}-\d{2}-\d{2})\s+Time: (\d{2}:\d{2}:\d{2})\s+(.*)', list_translation_frame.winfo_children()[0].cget('text'))        
        if match:
            date = match.group(1)
            time = match.group(2).replace(':', '-')
            sheet_name = date + '_' + time            
            try:
                workbook = openpyxl.load_workbook('history.xlsx')
                sheet_names = workbook.sheetnames
                if len(sheet_names) == 1:
                    os.remove('history.xlsx')
                    to_home_frame(history_frame, home_frame)
                    messagebox.showinfo('Notification', 'Your history is empty!')
                else:
                    sheet_to_remove = workbook[sheet_name]
                    workbook.remove(sheet_to_remove)
                    workbook.save('history.xlsx')
                    for translation in list_history_frame.winfo_children():
                        translation.destroy()
                    for translation in list_translation_frame.winfo_children():
                        translation.destroy()
                    if os.path.isfile('history.xlsx'):
                        translations = read_all_translations()
                        if len(translations) > 0:
                            for translation in translations:
                                translation_label = ctk.CTkLabel(master=list_history_frame, text=translation['sheet'], font=('Roboto', 15), anchor='w', justify='left')
                                translation_label.pack(fill='both')
                                translation_label.additional_info = translation['values']
                                translation_label.bind('<Button-1>', lambda event, param=translation_label.additional_info: history_translation_label_click(event, param))
                    messagebox.showinfo('Notification', 'Your history has been deleted!')
            except:
                messagebox.showerror('Error', 'Sorry, history deletion failed!')
        else:
            messagebox.showerror('Error', 'Sorry, history deletion failed!')
    else:
        messagebox.showinfo('Notification', 'Please select the translation on the list to deleted!')    

# =======================================================================================================================================================    

def check_add_dataset_information_frame():
    new_word = new_word_add_dataset_information_entry.get()
    if new_word == '':
        notification_add_dataset_information_label.configure(text=notification_add_dataset_information_0)
        return False
    else:
        words = read_all_words()
        if len(words) > 0:
            if new_word.title() in words:
                notification_add_dataset_information_label.configure(text=notification_add_dataset_information_1)
                return False
            else:
                return True
        else:
            return True

def check_update_dataset_retraining_information():
    new_word = choose_word_update_dataset_retraining_information_entry.get()
    if new_word == '':
        notification_update_dataset_retraining_information_label.configure(text=notification_update_dataset_retraining_information_0)
        return False
    else:
        words = read_all_words()
        if not new_word.title() in words:
            notification_update_dataset_retraining_information_label.configure(text=notification_update_dataset_retraining_information_1)
            return False
        else:
            return True

# =======================================================================================================================================================    

def prepare_to_add_dataset_information_frame():
    notification_add_dataset_information_label.configure(text=notification_add_dataset_information_0)
    new_word_add_dataset_information_entry.delete(0, 'end')
    for word in list_add_dataset_information_frame.winfo_children():
        word.destroy()
    words = read_all_words()
    if len(words) > 0:
        for word in words:
            word_label = ctk.CTkLabel(master=list_add_dataset_information_frame, text=word, font=('Roboto', 15), anchor='w', justify='left')
            word_label.pack(fill='both')
    else:
        word_label = ctk.CTkLabel(master=list_add_dataset_information_frame, text='No Data', font=('Roboto', 15), anchor='w', justify='left')
        word_label.pack(fill='both')

def prepare_to_add_dataset_holistic_frame():
    global count, add_dataset_holistic_state
    count = 0
    add_dataset_holistic_state = ''
    new_word = new_word_add_dataset_information_entry.get().title()
    add_dataset_holistic_word_label.configure(text=f'Word: {new_word}')
    add_dataset_holistic_notification_label.configure(text=notification_add_dataset_holistic_0)
    bone_add_dataset_holistic_switch.configure(variable=ctk.StringVar(value='off'))
    home_add_dataset_holistic_button.grid_forget()
    test_add_dataset_holistic_button.grid_forget()
    add_dataset_holistic_progress_bar.grid_forget()
    add_dataset_holistic_progress_label.grid_forget()
    start_add_dataset_holistic_button.grid(row=5, column=1, sticky='nsew', padx=5, pady=5)
    back_add_dataset_holistic_button.grid(row=6, column=1, sticky='nsew', padx=5, pady=5)

def prepare_to_update_dataset_rewording_frame():
    choose_word_update_dataset_rewording_entry.delete(0, 'end')
    new_word_update_dataset_rewording_entry.delete(0, 'end')
    for word in list_update_dataset_rewording_frame.winfo_children():
        word.destroy()
    words = read_all_words()
    if len(words) > 0:
        for word in words:
            word_label = ctk.CTkLabel(master=list_update_dataset_rewording_frame, text=word, font=('Roboto', 15), anchor='w', justify='left')
            word_label.pack(fill='both')
        return True
    else:
        messagebox.showwarning("Warning", "You must add datasets to use some of the features of this application!")
        return False

def prepare_to_update_dataset_retraining_information_frame():
    choose_word_update_dataset_retraining_information_entry.delete(0, 'end')
    for word in list_update_dataset_retraining_information_frame.winfo_children():
        word.destroy()
    words = read_all_words()
    if len(words) > 0:
        for word in words:
            word_label = ctk.CTkLabel(master=list_update_dataset_retraining_information_frame, text=word, font=('Roboto', 15), anchor='w', justify='left')
            word_label.pack(fill='both')
        return True        
    else:
        messagebox.showwarning("Warning", "You must add datasets to use some of the features of this application!")
        return False

def prepare_to_update_dataset_retraining_holistic_frame():
    global count, update_dataset_retraining_holistic_state
    count = 0
    update_dataset_retraining_holistic_state = ''
    new_word = choose_word_update_dataset_retraining_information_entry.get().title()
    update_dataset_retraining_holistic_word_label.configure(text=f'Word: {new_word}')
    update_dataset_retraining_holistic_notification_label.configure(text=notification_update_dataset_retraining_holistic_0)
    bone_update_dataset_retraining_holistic_switch.configure(variable=ctk.StringVar(value='off'))
    home_update_dataset_retraining_holistic_button.grid_forget()
    test_update_dataset_retraining_holistic_button.grid_forget()
    update_dataset_retraining_holistic_progress_bar.grid_forget()
    update_dataset_retraining_holistic_progress_label.grid_forget()
    start_update_dataset_retraining_holistic_button.grid(row=5, column=1, sticky='nsew', padx=5, pady=5)
    back_update_dataset_retraining_holistic_button.grid(row=6, column=1, sticky='nsew', padx=5, pady=5)

def prepare_to_delete_dataset_frame():
    choose_word_delete_dataset_entry.delete(0, 'end')
    for word in list_delete_dataset_frame.winfo_children():
        word.destroy()
    words = read_all_words()
    if len(words) > 0:
        for word in words:
            word_label = ctk.CTkLabel(master=list_delete_dataset_frame, text=word, font=('Roboto', 15), anchor='w', justify='left')
            word_label.pack(fill='both')
        return True    
    else:
        messagebox.showwarning("Warning", "You must add datasets to use some of the features of this application!")
        return False

def prepare_to_translate_frame():    
    words = read_all_words()
    if len(words) > 0:
        start_translate_button.grid(row=4, column=1, sticky='nsew', padx=5, pady=5)
        back_translate_button.grid(row=5, column=1, sticky='nsew', padx=5, pady=5)
        stop_translate_button.grid_forget()
        bone_translate_switch.grid(row=4, column=4, sticky='w', padx=5, pady=5)
        translate_label.grid_forget()
        translate_label.configure(text='')
        translate_progress_bar.grid_forget()
        translate_progress_bar.set(0)    
        for translation in list_translate_frame.winfo_children():
            translation.destroy()
        return True
    else:
        messagebox.showwarning("Warning", "You must add datasets to use some of the features of this application!")
        return False

def prepare_to_history_frame():
    for translation in list_history_frame.winfo_children():
        translation.destroy()
    for translation in list_translation_frame.winfo_children():
        translation.destroy()
    if os.path.isfile('history.xlsx'):
        translations = read_all_translations()
        if len(translations) > 0:
            for translation in translations:
                translation_label = ctk.CTkLabel(master=list_history_frame, text=translation['sheet'], font=('Roboto', 15), anchor='w', justify='left')
                translation_label.pack(fill='both')
                translation_label.additional_info = translation['values']
                translation_label.bind('<Button-1>', lambda event, param=translation_label.additional_info: history_translation_label_click(event, param))
        return True
    else:
        messagebox.showinfo("Notification", "There is no translation history list!")
        return False

# =======================================================================================================================================================    

def change_to_home_frame(initial_frame, destination_frame):
    global state, cap
    cap.release()
    app.title('In Sign (Home)')
    app.geometry('600x350')
    destination_frame.tkraise(initial_frame)
    state = 'home_frame'

def change_to_setting_frame(initial_frame, destination_frame):    
    global state, cap, pop_out_setting_state
    if not pop_out_setting_state:
        cap.release()
        app.title('In Sign (Setting)')
        app.geometry('450x730')
        destination_frame.tkraise(initial_frame)
        state = 'setting_frame'
    else:
        messagebox.showinfo("Notification", "You have opened the setting!")

def change_to_dataset_frame(initial_frame, destination_frame):
    global state, cap
    cap.release()
    app.title('In Sign (Dataset)')
    app.geometry('400x400')
    destination_frame.tkraise(initial_frame)
    state = 'dataset_frame'

def change_to_add_dataset_information_frame(initial_frame, destination_frame):    
    global state, cap
    cap.release()
    app.title('In Sign (Add Dataset)')
    app.geometry('600x400')
    destination_frame.tkraise(initial_frame)
    state = 'add_dataset_information_frame'
    prepare_to_add_dataset_information_frame()

def change_to_add_dataset_holistic_frame(initial_frame, destination_frame):
    global state, cap
    cap.release()
    cap = cv2.VideoCapture(0)
    app.title('In Sign (Add Dataset)')
    app.geometry('900x700')
    destination_frame.tkraise(initial_frame)
    state = 'add_dataset_holistic_frame'
    prepare_to_add_dataset_holistic_frame()
    add_dataset_holistic_camera()

def change_to_delete_dataset_frame(initial_frame, destination_frame):
    if prepare_to_delete_dataset_frame():
        global state, cap
        cap.release()
        app.title('In Sign (Delete Dataset)')
        app.geometry('600x400')
        destination_frame.tkraise(initial_frame)
        state = 'delete_dataset_frame'        

def change_to_update_dataset_frame(initial_frame, destination_frame):
    global state, cap
    cap.release()
    app.title('In Sign (Update Dataset)')
    app.geometry('400x290')
    destination_frame.tkraise(initial_frame)
    state = 'update_dataset_frame'

def change_to_update_dataset_rewording_frame(initial_frame, destination_frame):
    if prepare_to_update_dataset_rewording_frame():
        global state, cap
        cap.release()
        app.title('In Sign (Rewording Dataset)')
        app.geometry('600x450')
        destination_frame.tkraise(initial_frame)
        state = 'update_dataset_rewording_frame'

def change_to_update_dataset_retraining_information_frame(initial_frame, destination_frame):
    if prepare_to_update_dataset_retraining_information_frame():
        global state, cap
        cap.release()
        app.title('In Sign (Retraining Dataset)')
        app.geometry('600x400')
        destination_frame.tkraise(initial_frame)
        state = 'update_dataset_retraining_information_frame'

def change_to_update_dataset_retraining_holistic_frame(initial_frame, destination_frame):
    global state, cap
    cap.release()
    cap = cv2.VideoCapture(0)
    app.title('In Sign (Retraining Dataset)')
    app.geometry('900x700')
    destination_frame.tkraise(initial_frame)    
    state = 'update_dataset_retraining_holistic_frame'
    prepare_to_update_dataset_retraining_holistic_frame()
    update_dataset_retraining_holistic_camera()

def change_to_translate_frame(initial_frame, destination_frame):
    if prepare_to_translate_frame():
        global state, cap
        cap.release()
        cap = cv2.VideoCapture(0)
        app.title('In Sign (Translate)')
        app.geometry('1060x700')
        destination_frame.tkraise(initial_frame)    
        state = 'translate_frame'
        translate_camera()

def change_to_history_frame(initial_frame, destination_frame):
    if prepare_to_history_frame():
        global state, cap
        cap.release()
        app.title('In Sign (History)')
        app.geometry('800x600')
        destination_frame.tkraise(initial_frame)
        state = 'history_frame'

def change_to_tutorial_frame(initial_frame, destination_frame):
    global state, cap
    cap.release()
    app.title('In Sign (Tutorial)')
    app.geometry('700x600')
    destination_frame.tkraise(initial_frame)
    state = 'tutorial_frame'

# =======================================================================================================================================================    

def to_home_frame(initial_frame, destination_frame):
    global state
    if state == 'setting_frame':
        change_to_home_frame(initial_frame, destination_frame)
    elif state == 'dataset_frame':
        change_to_home_frame(initial_frame, destination_frame)
    elif state == 'add_dataset_holistic_frame':
        change_to_home_frame(initial_frame, destination_frame)
    elif state == 'update_dataset_retraining_holistic_frame':
        change_to_home_frame(initial_frame, destination_frame)
    elif state == 'translate_frame':
        change_to_home_frame(initial_frame, destination_frame)        
    elif state == 'history_frame':
        change_to_home_frame(initial_frame, destination_frame)
    elif state == 'tutorial_frame':
        change_to_home_frame(initial_frame, destination_frame)

def to_setting_frame(initial_frame, destination_frame):
    global state
    if state == 'home_frame':
        change_to_setting_frame(initial_frame, destination_frame)

def to_dataset_frame(initial_frame, destination_frame):
    global state
    if state == 'home_frame':
        change_to_dataset_frame(initial_frame, destination_frame)
    elif state == 'add_dataset_information_frame':
        change_to_dataset_frame(initial_frame, destination_frame)
    elif state == 'update_dataset_frame':
        change_to_dataset_frame(initial_frame, destination_frame)
    elif state == 'delete_dataset_frame':
        change_to_dataset_frame(initial_frame, destination_frame)

def to_add_dataset_information_frame(initial_frame, destination_frame):
    global state
    if state == 'dataset_frame':
        change_to_add_dataset_information_frame(initial_frame, destination_frame)
    elif state == 'add_dataset_holistic_frame':
        change_to_add_dataset_information_frame(initial_frame, destination_frame)

def to_add_dataset_holistic_frame(initial_frame, destination_frame):
    global state
    if state == 'add_dataset_information_frame':
        if check_add_dataset_information_frame():
            change_to_add_dataset_holistic_frame(initial_frame, destination_frame)

def to_delete_dataset_frame(initial_frame, destination_frame):
    global state
    if state == 'dataset_frame':
        change_to_delete_dataset_frame(initial_frame, destination_frame)

def to_update_dataset_frame(initial_frame, destination_frame):
    global state
    if state == 'dataset_frame':
        change_to_update_dataset_frame(initial_frame, destination_frame)
    elif state == 'update_dataset_rewording_frame':
        change_to_update_dataset_frame(initial_frame, destination_frame)
    elif state == 'update_dataset_retraining_information_frame':
        change_to_update_dataset_frame(initial_frame, destination_frame)

def to_update_dataset_rewording_frame(initial_frame, destination_frame):
    global state
    if state == 'update_dataset_frame':
        change_to_update_dataset_rewording_frame(initial_frame, destination_frame)

def to_update_dataset_retraining_information_frame(initial_frame, destination_frame):
    global state
    if state == 'update_dataset_frame':
        change_to_update_dataset_retraining_information_frame(initial_frame, destination_frame)
    elif state == 'update_dataset_retraining_holistic_frame':
        change_to_update_dataset_retraining_information_frame(initial_frame, destination_frame)

def to_update_dataset_retraining_holistic_frame(initial_frame, destination_frame):
    global state
    if state == 'update_dataset_retraining_information_frame':
        if check_update_dataset_retraining_information():
            change_to_update_dataset_retraining_holistic_frame(initial_frame, destination_frame)

def to_translate_frame(initial_frame, destination_frame):
    global state
    if state == 'home_frame':
        change_to_translate_frame(initial_frame, destination_frame)

def to_history_frame(initial_frame, destination_frame):
    global state
    if state == 'home_frame':
        change_to_history_frame(initial_frame, destination_frame)

def to_tutorial_frame(initial_frame, destination_frame):
    global state
    if state == 'home_frame':
        change_to_tutorial_frame(initial_frame, destination_frame)

def on_closing():
    global cap
    if add_dataset_holistic_state == 'start_button':
        pass
    elif update_dataset_retraining_holistic_state == 'start-button':
        pass
    elif translate_state == 'start-button':
        pass
    else:
        answer = messagebox.askquestion('Caution', 'Are you sure you want to quit?')
        if answer == 'yes':
            cap.release()
            app.destroy()

app.protocol("WM_DELETE_WINDOW", on_closing)

# =======================================================================================================================================================    

def set_appearance_mode_combobox(get_appearance_mode):
    ctk.set_appearance_mode(get_appearance_mode)

def set_color_theme_combobox(get_color_theme):
    if len(all_second_buttons) > 0 and len(all_second_switches) > 0:
        with open(os.path.join('D:/Project/InSign/venv/Lib/site-packages/customtkinter/assets/themes', f"{get_color_theme}.json"), "r") as file:
            theme_json = json.load(file)
        for button in all_second_buttons:
            button.configure(fg_color=theme_json['CTkButton']['fg_color'])
            button.configure(hover_color=theme_json['CTkButton']['hover_color'])
        for switch in all_second_switches:
            switch.configure(progress_color=theme_json['CTkButton']['fg_color'])
    if not training_buttons == None:
        with open(os.path.join('D:/Project/InSign/venv/Lib/site-packages/customtkinter/assets/themes', f"{get_color_theme}.json"), "r") as file:
            theme_json = json.load(file)
        training_buttons.configure(fg_color=theme_json['CTkButton']['fg_color'])
        training_buttons.configure(hover_color=theme_json['CTkButton']['hover_color'])
    with open(os.path.join('D:/Project/InSign/venv/Lib/site-packages/customtkinter/assets/themes', f"{get_color_theme}.json"), "r") as file:
        theme_json = json.load(file)
    for button in all_buttons:
        button.configure(fg_color=theme_json['CTkButton']['fg_color'])
        button.configure(hover_color=theme_json['CTkButton']['hover_color'])
    for switch in all_switches:
        switch.configure(progress_color=theme_json['CTkButton']['fg_color'])
    for slide in all_progress_bar:
        slide.configure(progress_color=theme_json['CTkButton']['fg_color'])

# =======================================================================================================================================================    

def on_closing_training_result_second_frame():
    global training_result_second_frame, training_buttons
    training_result_second_frame.destroy()
    training_result_second_frame = None
    training_buttons = None

def pop_out_training_result():
    global training_result_second_frame, training_buttons
    if training_result_second_frame == None:
        training_result_second_frame = ctk.CTkToplevel(app)
        training_result_second_frame.geometry('450x600')
        training_result_second_frame.geometry('+1200+50')
        training_result_second_frame.title('InSign (Training Result)')
        training_result_second_frame.resizable(width=False, height=False)
        training_result_second_frame.after(250, lambda: training_result_second_frame.iconbitmap('D:/Project/InSign/assets/InSign.ico'))

        training_result_second_frame.rowconfigure(0, minsize=50)
        training_result_second_frame.rowconfigure(1, minsize=80)
        training_result_second_frame.rowconfigure(2, weight=1)
        training_result_second_frame.rowconfigure(3, minsize=50)
        training_result_second_frame.rowconfigure(4, minsize=50)
        training_result_second_frame.columnconfigure(0, minsize=50)
        training_result_second_frame.columnconfigure(1, weight=1)
        training_result_second_frame.columnconfigure(2, minsize=50)

        training_result_second_frame_label = ctk.CTkLabel(master=training_result_second_frame, text="Training Result:", font=('Roboto', 25))
        training_result_second_frame_label.grid(row=1, column=1, sticky='n')

        training_result_second_frame_text = ctk.CTkTextbox(master=training_result_second_frame, wrap=ctk.WORD, font=('Roboto', 14), state='normal')
        training_result_second_frame_text.grid(row=2, column=1, sticky='nsew', padx=5, pady=5)

        close_training_result_second_button = ctk.CTkButton(master=training_result_second_frame, text='Close', font=('Roboto', 14), corner_radius=10, command=lambda: on_closing_training_result_second_frame())
        close_training_result_second_button.grid(row=3, column=1, sticky='nsew', padx=5, pady=5)

        training_buttons = close_training_result_second_button

        set_appearance_mode_combobox(appearance_mode_combobox.get())
        set_color_theme_combobox(color_theme_combobox.get())

        training_text = '''------------------------------------------------------------------------------------
Sorry, the dataset is empty!
Training cannot be carried out:(
------------------------------------------------------------------------------------
You can fill in the dataset via the add dataset feature!
Enjoy:)
------------------------------------------------------------------------------------
'''

        csv_file = open('dataset.csv', 'r', newline='')
        csv_reader = csv.reader(csv_file)
        row_count = sum(1 for row in csv_reader)
        if row_count > data_class:
            pass
            df = pd.read_csv('dataset.csv')
            X = df.drop('class', axis=1)
            y = df['class']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
            pipelines = {
                'rf': make_pipeline(StandardScaler(), RandomForestClassifier())
            }
            fit_models = {}
            for algo, pipeline in pipelines.items():
                model = pipeline.fit(X_train, y_train)
                fit_models[algo] = model
            fit_models['rf'].predict(X_test)            
            for algo, model in fit_models.items():
                y_predict = model.predict(X_test)
                training_text = f'''------------------------------------------------------------------------------------
algorithm: {algo}                                                                                 
------------------------------------------------------------------------------------

{classification_report(y_test, y_predict)}
------------------------------------------------------------------------------------
Data training has been updated!                     
Enjoy :)                                             
------------------------------------------------------------------------------------
'''            
            training_result_second_frame_text.insert(ctk.END, training_text)
            training_result_second_frame_text.tag_config("right", justify='right')
            training_result_second_frame_text.tag_add("right", "1.0", tk.END)
            training_result_second_frame_text.configure(state='disabled')
            with open('data_training.pkl', 'wb') as file:
                pickle.dump(fit_models['rf'], file)
        else:
            training_result_second_frame_text.insert(ctk.END, training_text)

        training_result_second_frame.protocol("WM_DELETE_WINDOW", on_closing_training_result_second_frame)
    else:
        training_result_second_frame.destroy()
        training_result_second_frame = None
        pop_out_training_result()

def synchronization_combobox_appearence_mode_setting(choice):
    appearance_mode_combobox.set(choice)
    set_appearance_mode_combobox(choice)

def synchronization_combobox_color_theme_setting(choice):
    color_theme_combobox.set(choice)
    set_color_theme_combobox(choice)

def synchronization_switch_setting():
    pose_landmark_switch.select() if pose_landmark_second_switch.get() == 'on' else pose_landmark_switch.deselect()
    face_landmark_switch.select() if face_landmark_second_switch.get() == 'on' else face_landmark_switch.deselect()
    left_landmark_switch.select() if left_landmark_second_switch.get() == 'on' else left_landmark_switch.deselect()
    right_landmark_switch.select() if right_landmark_second_switch.get() == 'on' else right_landmark_switch.deselect()
    face_rectangle_switch.select() if face_rectangle_second_switch.get() == 'on' else face_rectangle_switch.deselect()
    left_rectangle_switch.select() if left_rectangle_second_switch.get() == 'on' else left_rectangle_switch.deselect()
    right_rectangle_switch.select() if right_rectangle_second_switch.get() == 'on' else right_rectangle_switch.deselect()


def on_closing_setting_second_frame():
    global pop_out_setting_state, setting_second_frame, all_second_buttons, all_second_switches
    pop_out_setting_state = False
    setting_second_frame.destroy()
    all_second_buttons = []
    all_second_switches = []

def pop_out_setting_frame():
    global pop_out_setting_state, setting_second_frame
    global appearance_mode_second_combobox, color_theme_second_combobox, training_result_second_button, pose_landmark_second_switch, face_landmark_second_switch, left_landmark_second_switch, right_landmark_second_switch, left_rectangle_second_switch, right_rectangle_second_switch, face_rectangle_second_switch
    setting_second_frame = ctk.CTkToplevel(app)
    setting_second_frame.geometry('450x730')
    setting_second_frame.geometry('+900+50')
    setting_second_frame.title('InSign (Setting)')
    setting_second_frame.resizable(width=False, height=False)
    setting_second_frame.after(250, lambda: setting_second_frame.iconbitmap('D:/Project/InSign/assets/InSign.ico'))

    setting_second_frame.rowconfigure(0, minsize=50)
    setting_second_frame.rowconfigure(1, minsize=80)
    setting_second_frame.rowconfigure(2, weight=1)
    setting_second_frame.rowconfigure(3, weight=1)
    setting_second_frame.rowconfigure(4, weight=1)
    setting_second_frame.rowconfigure(5, weight=1)
    setting_second_frame.rowconfigure(6, weight=1)
    setting_second_frame.rowconfigure(7, weight=1)
    setting_second_frame.rowconfigure(8, weight=1)
    setting_second_frame.rowconfigure(9, weight=1)
    setting_second_frame.rowconfigure(10, weight=1)
    setting_second_frame.rowconfigure(11, minsize=50)
    setting_second_frame.rowconfigure(12, minsize=50)
    setting_second_frame.rowconfigure(13, minsize=50)
    setting_second_frame.columnconfigure(0, minsize=50)
    setting_second_frame.columnconfigure(1, weight=1)
    setting_second_frame.columnconfigure(2, weight=1)
    setting_second_frame.columnconfigure(3, minsize=50)

    setting_title_second_label = ctk.CTkLabel(master=setting_second_frame, text="Setting", font=('Roboto', 25))
    setting_title_second_label.grid(row=1, column=1, columnspan=2, sticky='n')

    appearance_mode_second_label = ctk.CTkLabel(master=setting_second_frame, text="Appearance Mode:", font=('Roboto', 14))
    appearance_mode_second_label.grid(row=2, column=1, sticky='w', padx=5)

    color_theme_second_label = ctk.CTkLabel(master=setting_second_frame, text="Color Theme:", font=('Roboto', 14))
    color_theme_second_label.grid(row=3, column=1, sticky='w', padx=5)

    appearance_mode_second_combobox = ctk.CTkComboBox(master=setting_second_frame, values=appearance_mode_values, font=('Roboto', 14), corner_radius=10, state='readonly', command=synchronization_combobox_appearence_mode_setting)
    appearance_mode_second_combobox.grid(row=2, column=2, sticky='ew', padx=5, pady=5)
    appearance_mode_second_combobox.set('system')

    color_theme_second_combobox = ctk.CTkComboBox(master=setting_second_frame, values=color_theme_values, font=('Roboto', 14), corner_radius=10, state='readonly', command=synchronization_combobox_color_theme_setting)
    color_theme_second_combobox.grid(row=3, column=2, sticky='ew', padx=5, pady=5)
    color_theme_second_combobox.set('green')    

    training_result_second_label = ctk.CTkLabel(master=setting_second_frame, text="Training Result:", font=('Roboto', 14))
    training_result_second_label.grid(row=11, column=1, sticky='w', padx=5)

    pose_landmark_second_label = ctk.CTkLabel(master=setting_second_frame, text="Pose Landmarks:", font=('Roboto', 14))
    pose_landmark_second_label.grid(row=4, column=1, sticky='w', padx=5)

    face_landmark_second_label = ctk.CTkLabel(master=setting_second_frame, text="Face Landmarks:", font=('Roboto', 14))
    face_landmark_second_label.grid(row=5, column=1, sticky='w', padx=5)

    left_landmark_second_label = ctk.CTkLabel(master=setting_second_frame, text="Left Hand Landmarks:", font=('Roboto', 14))
    left_landmark_second_label.grid(row=6, column=1, sticky='w', padx=5)

    right_landmark_second_label = ctk.CTkLabel(master=setting_second_frame, text="Right Hand Landmarks:", font=('Roboto', 14))
    right_landmark_second_label.grid(row=7, column=1, sticky='w', padx=5)

    face_rectangle_second_label = ctk.CTkLabel(master=setting_second_frame, text="Face Rectangle:", font=('Roboto', 14))
    face_rectangle_second_label.grid(row=8, column=1, sticky='w', padx=5)

    left_rectangle_second_label = ctk.CTkLabel(master=setting_second_frame, text="Left Hand Rectangle:", font=('Roboto', 14))
    left_rectangle_second_label.grid(row=9, column=1, sticky='w', padx=5)

    right_rectangle_second_label = ctk.CTkLabel(master=setting_second_frame, text="Right Hand Rectangle:", font=('Roboto', 14))
    right_rectangle_second_label.grid(row=10, column=1, sticky='w', padx=5)

    training_result_second_button = ctk.CTkButton(master=setting_second_frame, text='Open', font=('Roboto', 14), corner_radius=10, command=lambda: pop_out_training_result())
    training_result_second_button.grid(row=11, column=2, sticky='nsew', padx=5, pady=5)

    pose_landmark_second_switch = ctk.CTkSwitch(master=setting_second_frame, text='', variable=ctk.StringVar(value='off'), onvalue='on', offvalue='off', bg_color='transparent', font=('Roboto', 14), command=lambda: synchronization_switch_setting())
    pose_landmark_second_switch.grid(row=4, column=2, sticky='w', padx=5, pady=5)

    face_landmark_second_switch = ctk.CTkSwitch(master=setting_second_frame, text='', variable=ctk.StringVar(value='off'), onvalue='on', offvalue='off', bg_color='transparent', font=('Roboto', 14), command=lambda: synchronization_switch_setting())
    face_landmark_second_switch.grid(row=5, column=2, sticky='w', padx=5, pady=5)

    left_landmark_second_switch = ctk.CTkSwitch(master=setting_second_frame, text='', variable=ctk.StringVar(value='on'), onvalue='on', offvalue='off', bg_color='transparent', font=('Roboto', 14), command=lambda: synchronization_switch_setting())
    left_landmark_second_switch.grid(row=6, column=2, sticky='w', padx=5, pady=5)

    right_landmark_second_switch = ctk.CTkSwitch(master=setting_second_frame, text='', variable=ctk.StringVar(value='on'), onvalue='on', offvalue='off', bg_color='transparent', font=('Roboto', 14), command=lambda: synchronization_switch_setting())
    right_landmark_second_switch.grid(row=7, column=2, sticky='w', padx=5, pady=5)

    face_rectangle_second_switch = ctk.CTkSwitch(master=setting_second_frame, text='', variable=ctk.StringVar(value='off'), onvalue='on', offvalue='off', bg_color='transparent', font=('Roboto', 14), command=lambda: synchronization_switch_setting())
    face_rectangle_second_switch.grid(row=8, column=2, sticky='w', padx=5, pady=5)

    left_rectangle_second_switch = ctk.CTkSwitch(master=setting_second_frame, text='', variable=ctk.StringVar(value='off'), onvalue='on', offvalue='off', bg_color='transparent', font=('Roboto', 14), command=lambda: synchronization_switch_setting())
    left_rectangle_second_switch.grid(row=9, column=2, sticky='w', padx=5, pady=5)

    right_rectangle_second_switch = ctk.CTkSwitch(master=setting_second_frame, text='', variable=ctk.StringVar(value='off'), onvalue='on', offvalue='off', bg_color='transparent', font=('Roboto', 14), command=lambda: synchronization_switch_setting())
    right_rectangle_second_switch.grid(row=10, column=2, sticky='w', padx=5, pady=5)

    close_setting_second_button = ctk.CTkButton(master=setting_second_frame, text='Close', font=('Roboto', 14), corner_radius=10, command=lambda: on_closing_setting_second_frame())
    close_setting_second_button.grid(row=12, column=1, columnspan=2, sticky='nsew', padx=5, pady=5)

    appearance_mode_second_combobox.set(appearance_mode_combobox.get())
    color_theme_second_combobox.set(color_theme_combobox.get())

    pose_landmark_second_switch.select() if pose_landmark_switch.get() == 'on' else pose_landmark_second_switch.deselect()
    face_landmark_second_switch.select() if face_landmark_switch.get() == 'on' else face_landmark_second_switch.deselect()
    left_landmark_second_switch.select() if left_landmark_switch.get() == 'on' else left_landmark_second_switch.deselect()
    right_landmark_second_switch.select() if right_landmark_switch.get() == 'on' else right_landmark_second_switch.deselect()
    face_rectangle_second_switch.select() if face_rectangle_switch.get() == 'on' else face_rectangle_second_switch.deselect()
    left_rectangle_second_switch.select() if left_rectangle_switch.get() == 'on' else left_rectangle_second_switch.deselect()
    right_rectangle_second_switch.select() if right_rectangle_switch.get() == 'on' else right_rectangle_second_switch.deselect()

    all_second_buttons.extend([close_setting_second_button, training_result_second_button])
    all_second_switches.extend([right_rectangle_second_switch, left_rectangle_second_switch, face_rectangle_second_switch, right_landmark_second_switch, left_landmark_second_switch, pose_landmark_second_switch, face_landmark_second_switch])

    set_appearance_mode_combobox(appearance_mode_combobox.get())
    set_color_theme_combobox(color_theme_combobox.get())

    setting_second_frame.protocol("WM_DELETE_WINDOW", on_closing_setting_second_frame)
    pop_out_setting_state = True
    to_home_frame(setting_frame, home_frame)

# =======================================================================================================================================================

home_frame = ctk.CTkFrame(master=app, fg_color='transparent', corner_radius=0)
home_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=1, relheight=1)
home_frame.rowconfigure(0, minsize=50)
home_frame.rowconfigure(1, minsize=80)
home_frame.rowconfigure(2, weight=1)
home_frame.rowconfigure(3, weight=1)
home_frame.rowconfigure(4, minsize=50)
home_frame.columnconfigure(0, minsize=50)
home_frame.columnconfigure(1, weight=1)
home_frame.columnconfigure(2, weight=1)
home_frame.columnconfigure(3, weight=1)
home_frame.columnconfigure(4, minsize=50)

# =======================================================================================================================================================

setting_frame = ctk.CTkFrame(master=app, fg_color='transparent', corner_radius=0)
setting_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=1, relheight=1)
setting_frame.rowconfigure(0, minsize=50)
setting_frame.rowconfigure(1, minsize=80)
setting_frame.rowconfigure(2, weight=1)
setting_frame.rowconfigure(3, weight=1)
setting_frame.rowconfigure(4, weight=1)
setting_frame.rowconfigure(5, weight=1)
setting_frame.rowconfigure(6, weight=1)
setting_frame.rowconfigure(7, weight=1)
setting_frame.rowconfigure(8, weight=1)
setting_frame.rowconfigure(9, weight=1)
setting_frame.rowconfigure(10, weight=1)
setting_frame.rowconfigure(11, minsize=50)
setting_frame.rowconfigure(12, minsize=50)
setting_frame.rowconfigure(13, minsize=50)
setting_frame.columnconfigure(0, minsize=50)
setting_frame.columnconfigure(1, weight=1)
setting_frame.columnconfigure(2, weight=1)
setting_frame.columnconfigure(3, minsize=50)

# =======================================================================================================================================================

dataset_frame = ctk.CTkFrame(master=app, fg_color='transparent', corner_radius=0)
dataset_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=1, relheight=1)
dataset_frame.rowconfigure(0, minsize=50)
dataset_frame.rowconfigure(1, minsize=80)
dataset_frame.rowconfigure(2, weight=1)
dataset_frame.rowconfigure(3, weight=1)
dataset_frame.rowconfigure(4, minsize=50)
dataset_frame.rowconfigure(5, minsize=50)
dataset_frame.columnconfigure(0, minsize=50)
dataset_frame.columnconfigure(1, weight=1)
dataset_frame.columnconfigure(2, weight=1)
dataset_frame.columnconfigure(3, minsize=50)

# =======================================================================================================================================================

add_dataset_information_frame = ctk.CTkFrame(master=app, fg_color='transparent', corner_radius=0)
add_dataset_information_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=1, relheight=1)
add_dataset_information_frame.rowconfigure(0, minsize=50)
add_dataset_information_frame.rowconfigure(1, minsize=80)
add_dataset_information_frame.rowconfigure(2, minsize=50)
add_dataset_information_frame.rowconfigure(3, minsize=50)
add_dataset_information_frame.rowconfigure(4, weight=1)
add_dataset_information_frame.rowconfigure(5, minsize=50)
add_dataset_information_frame.rowconfigure(6, minsize=50)
add_dataset_information_frame.columnconfigure(0, minsize=50)
add_dataset_information_frame.columnconfigure(1, weight=1)
add_dataset_information_frame.columnconfigure(2, weight=1)
add_dataset_information_frame.columnconfigure(3, minsize=50)

# =======================================================================================================================================================

add_dataset_holistic_frame = ctk.CTkFrame(master=app, fg_color='transparent', corner_radius=0)
add_dataset_holistic_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=1, relheight=1)
add_dataset_holistic_frame.rowconfigure(0, minsize=50)
add_dataset_holistic_frame.rowconfigure(1, minsize=80)
add_dataset_holistic_frame.rowconfigure(2, minsize=50)
add_dataset_holistic_frame.rowconfigure(3, weight=2)
add_dataset_holistic_frame.rowconfigure(4, weight=1)
add_dataset_holistic_frame.rowconfigure(5, minsize=50)
add_dataset_holistic_frame.rowconfigure(6, minsize=50)
add_dataset_holistic_frame.rowconfigure(7, minsize=50)
add_dataset_holistic_frame.columnconfigure(0, minsize=50)
add_dataset_holistic_frame.columnconfigure(1, minsize=200)
add_dataset_holistic_frame.columnconfigure(2, weight=1)
add_dataset_holistic_frame.columnconfigure(3, minsize=200)
add_dataset_holistic_frame.columnconfigure(4, minsize=50)

# =======================================================================================================================================================

delete_dataset_frame = ctk.CTkFrame(master=app, fg_color='transparent', corner_radius=0)
delete_dataset_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=1, relheight=1)
delete_dataset_frame.rowconfigure(0, minsize=50)
delete_dataset_frame.rowconfigure(1, minsize=80)
delete_dataset_frame.rowconfigure(2, minsize=50)
delete_dataset_frame.rowconfigure(3, minsize=50)
delete_dataset_frame.rowconfigure(4, weight=1)
delete_dataset_frame.rowconfigure(5, minsize=50)
delete_dataset_frame.rowconfigure(6, minsize=50)
delete_dataset_frame.columnconfigure(0, minsize=50)
delete_dataset_frame.columnconfigure(1, weight=1)
delete_dataset_frame.columnconfigure(2, weight=1)
delete_dataset_frame.columnconfigure(3, minsize=50)

# =======================================================================================================================================================

update_dataset_frame = ctk.CTkFrame(master=app, fg_color='transparent', corner_radius=0)
update_dataset_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=1, relheight=1)
update_dataset_frame.rowconfigure(0, minsize=50)
update_dataset_frame.rowconfigure(1, minsize=80)
update_dataset_frame.rowconfigure(2, weight=1)
update_dataset_frame.rowconfigure(3, minsize=50)
update_dataset_frame.rowconfigure(4, minsize=50)
update_dataset_frame.columnconfigure(0, minsize=50)
update_dataset_frame.columnconfigure(1, weight=1)
update_dataset_frame.columnconfigure(2, weight=1)
update_dataset_frame.columnconfigure(3, minsize=50)

# =======================================================================================================================================================

update_dataset_rewording_frame = ctk.CTkFrame(master=app, fg_color='transparent', corner_radius=0)
update_dataset_rewording_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=1, relheight=1)
update_dataset_rewording_frame.rowconfigure(0, minsize=50)
update_dataset_rewording_frame.rowconfigure(1, minsize=80)
update_dataset_rewording_frame.rowconfigure(2, minsize=50)
update_dataset_rewording_frame.rowconfigure(3, minsize=50)
update_dataset_rewording_frame.rowconfigure(4, minsize=50)
update_dataset_rewording_frame.rowconfigure(5, minsize=50)
update_dataset_rewording_frame.rowconfigure(6, weight=1)
update_dataset_rewording_frame.rowconfigure(7, minsize=50)
update_dataset_rewording_frame.rowconfigure(8, minsize=50)
update_dataset_rewording_frame.columnconfigure(0, minsize=50)
update_dataset_rewording_frame.columnconfigure(1, weight=1)
update_dataset_rewording_frame.columnconfigure(2, weight=1)
update_dataset_rewording_frame.columnconfigure(3, minsize=50)

# =======================================================================================================================================================

update_dataset_retraining_information_frame = ctk.CTkFrame(master=app, fg_color='transparent', corner_radius=0)
update_dataset_retraining_information_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=1, relheight=1)
update_dataset_retraining_information_frame.rowconfigure(0, minsize=50)
update_dataset_retraining_information_frame.rowconfigure(1, minsize=80)
update_dataset_retraining_information_frame.rowconfigure(2, minsize=50)
update_dataset_retraining_information_frame.rowconfigure(3, minsize=50)
update_dataset_retraining_information_frame.rowconfigure(4, weight=1)
update_dataset_retraining_information_frame.rowconfigure(5, minsize=50)
update_dataset_retraining_information_frame.rowconfigure(6, minsize=50)
update_dataset_retraining_information_frame.columnconfigure(0, minsize=50)
update_dataset_retraining_information_frame.columnconfigure(1, weight=1)
update_dataset_retraining_information_frame.columnconfigure(2, weight=1)
update_dataset_retraining_information_frame.columnconfigure(3, minsize=50)

# =======================================================================================================================================================

update_dataset_retraining_holistic_frame = ctk.CTkFrame(master=app, fg_color='transparent', corner_radius=0)
update_dataset_retraining_holistic_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=1, relheight=1)
update_dataset_retraining_holistic_frame.rowconfigure(0, minsize=50)
update_dataset_retraining_holistic_frame.rowconfigure(1, minsize=80)
update_dataset_retraining_holistic_frame.rowconfigure(2, minsize=50)
update_dataset_retraining_holistic_frame.rowconfigure(3, weight=2)
update_dataset_retraining_holistic_frame.rowconfigure(4, weight=1)
update_dataset_retraining_holistic_frame.rowconfigure(5, minsize=50)
update_dataset_retraining_holistic_frame.rowconfigure(6, minsize=50)
update_dataset_retraining_holistic_frame.rowconfigure(7, minsize=50)
update_dataset_retraining_holistic_frame.columnconfigure(0, minsize=50)
update_dataset_retraining_holistic_frame.columnconfigure(1, minsize=200)
update_dataset_retraining_holistic_frame.columnconfigure(2, weight=1)
update_dataset_retraining_holistic_frame.columnconfigure(3, minsize=200)
update_dataset_retraining_holistic_frame.columnconfigure(4, minsize=50)

# =======================================================================================================================================================

translate_frame = ctk.CTkFrame(master=app, fg_color='transparent', corner_radius=0)
translate_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=1, relheight=1)
translate_frame.rowconfigure(0, minsize=50)
translate_frame.rowconfigure(1, minsize=80)
translate_frame.rowconfigure(2, minsize=50)
translate_frame.rowconfigure(3, weight=1)
translate_frame.rowconfigure(4, minsize=50)
translate_frame.rowconfigure(5, minsize=50)
translate_frame.rowconfigure(6, minsize=50)
translate_frame.columnconfigure(0, minsize=50)
translate_frame.columnconfigure(1, minsize=200)
translate_frame.columnconfigure(2, weight=1)
translate_frame.columnconfigure(3, minsize=200)
translate_frame.columnconfigure(4, minsize=200)
translate_frame.columnconfigure(5, minsize=50)

# =======================================================================================================================================================

tutorial_frame = ctk.CTkFrame(master=app, fg_color='transparent', corner_radius=0)
tutorial_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=1, relheight=1)
tutorial_frame.rowconfigure(0, minsize=50)
tutorial_frame.rowconfigure(1, minsize=80)
tutorial_frame.rowconfigure(2, weight=1)
tutorial_frame.rowconfigure(3, minsize=50)
tutorial_frame.rowconfigure(4, minsize=50)
tutorial_frame.columnconfigure(0, minsize=50)
tutorial_frame.columnconfigure(1, weight=1)
tutorial_frame.columnconfigure(2, minsize=50)

# =======================================================================================================================================================

history_frame = ctk.CTkFrame(master=app, fg_color='transparent', corner_radius=0)
history_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=1, relheight=1)
history_frame.rowconfigure(0, minsize=50)
history_frame.rowconfigure(1, minsize=80)
history_frame.rowconfigure(2, minsize=50)
history_frame.rowconfigure(3, weight=1)
history_frame.rowconfigure(4, minsize=50)
history_frame.rowconfigure(5, minsize=50)
history_frame.columnconfigure(0, minsize=50)
history_frame.columnconfigure(1, minsize=200)
history_frame.columnconfigure(2, weight=1)
history_frame.columnconfigure(3, weight=1)
history_frame.columnconfigure(4, minsize=200)
history_frame.columnconfigure(5, minsize=50)

# =======================================================================================================================================================

# home frame

home_title_label = ctk.CTkLabel(master=home_frame, text="Welcome, let's do sign language!", font=('Roboto', 25))
home_title_label.grid(row=1, column=1, columnspan=3, sticky='n')

dataset_button = ctk.CTkButton(master=home_frame, text='Dataset', font=('Roboto', 14), corner_radius=10, command=lambda: to_dataset_frame(home_frame, dataset_frame))
dataset_button.grid(row=2, column=1, sticky='nsew', padx=5, pady=5)

setting_button = ctk.CTkButton(master=home_frame, text='Setting', font=('Roboto', 14), corner_radius=10, command=lambda: to_setting_frame(home_frame, setting_frame))
setting_button.grid(row=3, column=2, sticky='nsew', padx=5, pady=5)

translate_button = ctk.CTkButton(master=home_frame, text='Translate', font=('Roboto', 14), corner_radius=10, command=lambda: to_translate_frame(home_frame, translate_frame))
translate_button.grid(row=2, column=3, rowspan=2, sticky='nsew', padx=5, pady=5)

history_button = ctk.CTkButton(master=home_frame, text='History', font=('Roboto', 14), corner_radius=10, command=lambda: to_history_frame(home_frame, history_frame))
history_button.grid(row=2, column=2, sticky='nsew', padx=5, pady=5)

tutorial_button = ctk.CTkButton(master=home_frame, text='Tutorial', font=('Roboto', 14), corner_radius=10, command=lambda: to_tutorial_frame(home_frame, tutorial_frame))
tutorial_button.grid(row=3, column=1, sticky='nsew', padx=5, pady=5)

all_buttons.extend([dataset_button, setting_button, translate_button, history_button, tutorial_button])

# =======================================================================================================================================================

# setting frame

setting_title_label = ctk.CTkLabel(master=setting_frame, text="Setting", font=('Roboto', 25))
setting_title_label.grid(row=1, column=1, columnspan=2, sticky='n')

appearance_mode_label = ctk.CTkLabel(master=setting_frame, text="Appearance Mode:", font=('Roboto', 14))
appearance_mode_label.grid(row=2, column=1, sticky='w', padx=5)

color_theme_label = ctk.CTkLabel(master=setting_frame, text="Color Theme:", font=('Roboto', 14))
color_theme_label.grid(row=3, column=1, sticky='w', padx=5)

appearance_mode_combobox = ctk.CTkComboBox(master=setting_frame, values=appearance_mode_values, font=('Roboto', 14), corner_radius=10, state='readonly', command=set_appearance_mode_combobox)
appearance_mode_combobox.grid(row=2, column=2, sticky='ew', padx=5, pady=5)
appearance_mode_combobox.set('system')

color_theme_combobox = ctk.CTkComboBox(master=setting_frame, values=color_theme_values, font=('Roboto', 14), corner_radius=10, state='readonly', command=set_color_theme_combobox)
color_theme_combobox.grid(row=3, column=2, sticky='ew', padx=5, pady=5)
color_theme_combobox.set('green')

training_result_label = ctk.CTkLabel(master=setting_frame, text="Training Result:", font=('Roboto', 14))
training_result_label.grid(row=11, column=1, sticky='w', padx=5)

pose_landmark_label = ctk.CTkLabel(master=setting_frame, text="Pose Landmarks:", font=('Roboto', 14))
pose_landmark_label.grid(row=4, column=1, sticky='w', padx=5)

face_landmark_label = ctk.CTkLabel(master=setting_frame, text="Face Landmarks:", font=('Roboto', 14))
face_landmark_label.grid(row=5, column=1, sticky='w', padx=5)

left_landmark_label = ctk.CTkLabel(master=setting_frame, text="Left Hand Landmarks:", font=('Roboto', 14))
left_landmark_label.grid(row=6, column=1, sticky='w', padx=5)

right_landmark_label = ctk.CTkLabel(master=setting_frame, text="Right Hand Landmarks:", font=('Roboto', 14))
right_landmark_label.grid(row=7, column=1, sticky='w', padx=5)

face_rectangle_label = ctk.CTkLabel(master=setting_frame, text="Face Rectangle:", font=('Roboto', 14))
face_rectangle_label.grid(row=8, column=1, sticky='w', padx=5)

left_rectangle_label = ctk.CTkLabel(master=setting_frame, text="Left Hand Rectangle:", font=('Roboto', 14))
left_rectangle_label.grid(row=9, column=1, sticky='w', padx=5)

right_rectangle_label = ctk.CTkLabel(master=setting_frame, text="Right Hand Rectangle:", font=('Roboto', 14))
right_rectangle_label.grid(row=10, column=1, sticky='w', padx=5)

training_result_button = ctk.CTkButton(master=setting_frame, text='Open', font=('Roboto', 14), corner_radius=10, command=lambda: pop_out_training_result())
training_result_button.grid(row=11, column=2, sticky='nsew', padx=5, pady=5)

pose_landmark_switch = ctk.CTkSwitch(master=setting_frame, text='', variable=ctk.StringVar(value='off'), onvalue='on', offvalue='off', bg_color='transparent', font=('Roboto', 14))
pose_landmark_switch.grid(row=4, column=2, sticky='w', padx=5, pady=5)

face_landmark_switch = ctk.CTkSwitch(master=setting_frame, text='', variable=ctk.StringVar(value='off'), onvalue='on', offvalue='off', bg_color='transparent', font=('Roboto', 14))
face_landmark_switch.grid(row=5, column=2, sticky='w', padx=5, pady=5)

left_landmark_switch = ctk.CTkSwitch(master=setting_frame, text='', variable=ctk.StringVar(value='on'), onvalue='on', offvalue='off', bg_color='transparent', font=('Roboto', 14))
left_landmark_switch.grid(row=6, column=2, sticky='w', padx=5, pady=5)

right_landmark_switch = ctk.CTkSwitch(master=setting_frame, text='', variable=ctk.StringVar(value='on'), onvalue='on', offvalue='off', bg_color='transparent', font=('Roboto', 14))
right_landmark_switch.grid(row=7, column=2, sticky='w', padx=5, pady=5)

face_rectangle_switch = ctk.CTkSwitch(master=setting_frame, text='', variable=ctk.StringVar(value='off'), onvalue='on', offvalue='off', bg_color='transparent', font=('Roboto', 14))
face_rectangle_switch.grid(row=8, column=2, sticky='w', padx=5, pady=5)

left_rectangle_switch = ctk.CTkSwitch(master=setting_frame, text='', variable=ctk.StringVar(value='off'), onvalue='on', offvalue='off', bg_color='transparent', font=('Roboto', 14))
left_rectangle_switch.grid(row=9, column=2, sticky='w', padx=5, pady=5)

right_rectangle_switch = ctk.CTkSwitch(master=setting_frame, text='', variable=ctk.StringVar(value='off'), onvalue='on', offvalue='off', bg_color='transparent', font=('Roboto', 14))
right_rectangle_switch.grid(row=10, column=2, sticky='w', padx=5, pady=5)

pop_out_setting_button = ctk.CTkButton(master=setting_frame, text='Pop Out', font=('Roboto', 14), corner_radius=10, command=lambda: pop_out_setting_frame())
pop_out_setting_button.grid(row=12, column=2, sticky='nsew', padx=5, pady=5)

back_setting_button = ctk.CTkButton(master=setting_frame, text='Back', font=('Roboto', 14), corner_radius=10, command=lambda: to_home_frame(setting_frame, home_frame))
back_setting_button.grid(row=12, column=1, sticky='nsew', padx=5, pady=5)

all_buttons.extend([back_setting_button, pop_out_setting_button, training_result_button])
all_switches.extend([right_rectangle_switch, left_rectangle_switch, face_rectangle_switch, right_landmark_switch, left_landmark_switch, pose_landmark_switch])

# =======================================================================================================================================================

# dataset frame

dataset_title_label = ctk.CTkLabel(master=dataset_frame, text="Dataset", font=('Roboto', 25))
dataset_title_label.grid(row=1, column=1, columnspan=2, sticky='n')

update_dataset_button = ctk.CTkButton(master=dataset_frame, text='Update', font=('Roboto', 14), corner_radius=10, command=lambda: to_update_dataset_frame(dataset_frame, update_dataset_frame))
update_dataset_button.grid(row=2, column=1, sticky='nsew', padx=5, pady=5)

delete_dataset_button = ctk.CTkButton(master=dataset_frame, text='Delete', font=('Roboto', 14), corner_radius=10, command=lambda: to_delete_dataset_frame(dataset_frame, delete_dataset_frame))
delete_dataset_button.grid(row=3, column=1, sticky='nsew', padx=5, pady=5)

add_dataset_button = ctk.CTkButton(master=dataset_frame, text='Add', font=('Roboto', 14), corner_radius=10, command=lambda: to_add_dataset_information_frame(dataset_frame, add_dataset_information_frame))
add_dataset_button.grid(row=2, column=2, rowspan=2, sticky='nsew', padx=5, pady=5)

back_dataset_button = ctk.CTkButton(master=dataset_frame, text='Back', font=('Roboto', 14), corner_radius=10, command=lambda: to_home_frame(dataset_frame, home_frame))
back_dataset_button.grid(row=4, column=1, columnspan=2, sticky='nsew', padx=5, pady=5)

all_buttons.extend([update_dataset_button, delete_dataset_button, add_dataset_button, back_dataset_button])

# =======================================================================================================================================================

# add dataset information

add_dataset_information_title_label = ctk.CTkLabel(master=add_dataset_information_frame, text="Add Dataset", font=('Roboto', 25))
add_dataset_information_title_label.grid(row=1, column=1, columnspan=2, sticky='n')

list_add_dataset_information_label = ctk.CTkLabel(master=add_dataset_information_frame, text="List of Words:", font=('Roboto', 14))
list_add_dataset_information_label.grid(row=2, column=1, sticky='w', padx=5)

list_add_dataset_information_frame = ctk.CTkScrollableFrame(master=add_dataset_information_frame, corner_radius=10, orientation='vertical', width=145)
list_add_dataset_information_frame.grid(row=3, column=1, rowspan=2, sticky='nsew', padx=5, pady=5)

new_word_add_dataset_information_label = ctk.CTkLabel(master=add_dataset_information_frame, text="New Word:", font=('Roboto', 14))
new_word_add_dataset_information_label.grid(row=2, column=2, sticky='w', padx=5)

new_word_add_dataset_information_entry = ctk.CTkEntry(master=add_dataset_information_frame, corner_radius=10, font=("Roboto", 15))
new_word_add_dataset_information_entry.grid(row=3, column=2, sticky='nsew', padx=5, pady=5)

notification_add_dataset_information_label = ctk.CTkLabel(master=add_dataset_information_frame, text=notification_add_dataset_information_0, font=('Roboto', 14), wraplength=200, anchor='center')
notification_add_dataset_information_label.grid(row=4, column=2, sticky='nsew', padx=5)

back_add_dataset_information_button = ctk.CTkButton(master=add_dataset_information_frame, text='Back', font=('Roboto', 14), corner_radius=10, command=lambda: to_dataset_frame(add_dataset_information_frame, dataset_frame))
back_add_dataset_information_button.grid(row=5, column=1, sticky='nsew', padx=5, pady=5)

next_add_dataset_information_button = ctk.CTkButton(master=add_dataset_information_frame, text='Next', font=('Roboto', 14), corner_radius=10, command=lambda: to_add_dataset_holistic_frame(add_dataset_information_frame, add_dataset_holistic_frame))
next_add_dataset_information_button.grid(row=5, column=2, sticky='nsew', padx=5, pady=5)

all_buttons.extend([back_add_dataset_information_button, next_add_dataset_information_button])

# =======================================================================================================================================================

# add dataset holistic

add_dataset_holistic_title_label = ctk.CTkLabel(master=add_dataset_holistic_frame, text="Add Dataset", font=('Roboto', 25))
add_dataset_holistic_title_label.grid(row=1, column=1, columnspan=3, sticky='n')

add_dataset_holistic_camera_label = tk.Label(master=add_dataset_holistic_frame, font=('Roboto', 25))
add_dataset_holistic_camera_label.grid(row=2, column=1, columnspan=2, rowspan=3, sticky='nsew', padx=5, pady=5)

add_dataset_holistic_word_label = ctk.CTkLabel(master=add_dataset_holistic_frame, text='Word: Word', font=('Roboto', 14), anchor='w')
add_dataset_holistic_word_label.grid(row=2, column=3, sticky='nsew', padx=5, pady=5)

instructions_add_dataset_holistic = '''Instructions:

1. Click the start button to get started!

2. Only one person in front of the camera!

3. Do sign language!
'''

add_dataset_holistic_instruction_label = ctk.CTkLabel(master=add_dataset_holistic_frame, text=instructions_add_dataset_holistic, font=('Roboto', 14), anchor='nw', justify='left', wraplength=150)
add_dataset_holistic_instruction_label.grid(row=3, column=3, sticky='nsew', padx=5, pady=5)

add_dataset_holistic_notification_label = ctk.CTkLabel(master=add_dataset_holistic_frame, text=notification_add_dataset_holistic_0, font=('Roboto', 14), anchor='nw', justify='left', wraplength=150)
add_dataset_holistic_notification_label.grid(row=4, column=3, sticky='nsew', padx=5, pady=5)

start_add_dataset_holistic_button = ctk.CTkButton(master=add_dataset_holistic_frame, text='Start', font=('Roboto', 14), corner_radius=10, command=lambda: add_dataset_holistic_start_button_click())
start_add_dataset_holistic_button.grid(row=5, column=1, sticky='nsew', padx=5, pady=5)

back_add_dataset_holistic_button = ctk.CTkButton(master=add_dataset_holistic_frame, text='Back', font=('Roboto', 14), corner_radius=10, command=lambda: to_add_dataset_information_frame(add_dataset_holistic_frame, add_dataset_information_frame))
back_add_dataset_holistic_button.grid(row=6, column=1, sticky='nsew', padx=5, pady=5)

test_add_dataset_holistic_button = ctk.CTkButton(master=add_dataset_holistic_frame, text='Testing', font=('Roboto', 14), corner_radius=10, command=lambda: add_dataset_holistic_test_button_click())
test_add_dataset_holistic_button.grid(row=6, column=3, sticky='nsew', padx=5, pady=5)

bone_add_dataset_holistic_switch = ctk.CTkSwitch(master=add_dataset_holistic_frame, text='Bone', variable=ctk.StringVar(value='off'), onvalue='on', offvalue='off', bg_color='transparent', font=('Roboto', 14))
bone_add_dataset_holistic_switch.grid(row=5, column=3, sticky='w', padx=5, pady=5)

home_add_dataset_holistic_button = ctk.CTkButton(master=add_dataset_holistic_frame, text='Home', font=('Roboto', 14), corner_radius=10, command=lambda: to_home_frame(add_dataset_holistic_frame, home_frame))
home_add_dataset_holistic_button.grid(row=5, column=1, sticky='nsew', padx=5, pady=5)

add_dataset_holistic_progress_bar = ctk.CTkProgressBar(master=add_dataset_holistic_frame, orientation='horizontal')
add_dataset_holistic_progress_bar.grid(row=5, column=2, sticky='ew', padx=5, pady=5)
add_dataset_holistic_progress_bar.set(0)

add_dataset_holistic_progress_label = ctk.CTkLabel(master=add_dataset_holistic_frame, text='0%', font=('Roboto', 14))
add_dataset_holistic_progress_label.grid(row=6, column=2, sticky='nsew', padx=5, pady=5)

all_buttons.extend([start_add_dataset_holistic_button, back_add_dataset_holistic_button, test_add_dataset_holistic_button, home_add_dataset_holistic_button])
all_switches.append(bone_add_dataset_holistic_switch)
all_progress_bar.append(add_dataset_holistic_progress_bar)

# =======================================================================================================================================================

# update dataset frame

update_dataset_title_label = ctk.CTkLabel(master=update_dataset_frame, text="Update Dataset", font=('Roboto', 25))
update_dataset_title_label.grid(row=1, column=1, columnspan=2, sticky='n')

update_dataset_rewording_button = ctk.CTkButton(master=update_dataset_frame, text='Rewording', font=('Roboto', 14), corner_radius=10, command=lambda: to_update_dataset_rewording_frame(update_dataset_frame, update_dataset_rewording_frame))
update_dataset_rewording_button.grid(row=2, column=1, sticky='nsew', padx=5, pady=5)

update_dataset_retraining_button = ctk.CTkButton(master=update_dataset_frame, text='Retraining', font=('Roboto', 14), corner_radius=10, command=lambda: to_update_dataset_retraining_information_frame(update_dataset_frame, update_dataset_retraining_information_frame))
update_dataset_retraining_button.grid(row=2, column=2, sticky='nsew', padx=5, pady=5)

back_update_dataset_button = ctk.CTkButton(master=update_dataset_frame, text='Back', font=('Roboto', 14), corner_radius=10, command=lambda: to_dataset_frame(update_dataset_frame, dataset_frame))
back_update_dataset_button.grid(row=3, column=1, columnspan=2, sticky='nsew', padx=5, pady=5)

all_buttons.extend([update_dataset_rewording_button, update_dataset_retraining_button, back_update_dataset_button])

# =======================================================================================================================================================

# update dataset rewording frame

update_dataset_rewording_title_label = ctk.CTkLabel(master=update_dataset_rewording_frame, text="Rewording Dataset", font=('Roboto', 25))
update_dataset_rewording_title_label.grid(row=1, column=1, columnspan=2, sticky='n')

list_update_dataset_rewording_label = ctk.CTkLabel(master=update_dataset_rewording_frame, text="List of Words:", font=('Roboto', 14))
list_update_dataset_rewording_label.grid(row=2, column=1, sticky='w', padx=5)

list_update_dataset_rewording_frame = ctk.CTkScrollableFrame(master=update_dataset_rewording_frame, corner_radius=10, orientation='vertical', width=145)
list_update_dataset_rewording_frame.grid(row=3, column=1, rowspan=4, sticky='nsew', padx=5, pady=5)

choose_word_update_dataset_rewording_label = ctk.CTkLabel(master=update_dataset_rewording_frame, text="Choose a Word:", font=('Roboto', 14))
choose_word_update_dataset_rewording_label.grid(row=2, column=2, sticky='w', padx=5)

choose_word_update_dataset_rewording_entry = ctk.CTkEntry(master=update_dataset_rewording_frame, corner_radius=10, font=("Roboto", 15))
choose_word_update_dataset_rewording_entry.grid(row=3, column=2, sticky='nsew', padx=5, pady=5)

new_word_update_dataset_rewording_label = ctk.CTkLabel(master=update_dataset_rewording_frame, text="New Word:", font=('Roboto', 14))
new_word_update_dataset_rewording_label.grid(row=4, column=2, sticky='w', padx=5)

new_word_update_dataset_rewording_entry = ctk.CTkEntry(master=update_dataset_rewording_frame, corner_radius=10, font=("Roboto", 15))
new_word_update_dataset_rewording_entry.grid(row=5, column=2, sticky='nsew', padx=5, pady=5)

notification_update_dataset_rewording_label = ctk.CTkLabel(master=update_dataset_rewording_frame, text=notification_update_dataset_rewording_0, font=('Roboto', 14), wraplength=200, anchor='center')
notification_update_dataset_rewording_label.grid(row=6, column=2, sticky='nsew', padx=5)

back_update_dataset_rewording_button = ctk.CTkButton(master=update_dataset_rewording_frame, text='Back', font=('Roboto', 14), corner_radius=10, command=lambda: to_update_dataset_frame(update_dataset_rewording_frame, update_dataset_frame))
back_update_dataset_rewording_button.grid(row=7, column=1, sticky='nsew', padx=5, pady=5)

agree_update_dataset_rewording_button = ctk.CTkButton(master=update_dataset_rewording_frame, text='Rewording', font=('Roboto', 14), corner_radius=10, command=lambda: update_dataset_rewording_button_click())
agree_update_dataset_rewording_button.grid(row=7, column=2, sticky='nsew', padx=5, pady=5)

all_buttons.extend([back_update_dataset_rewording_button, agree_update_dataset_rewording_button])

# =======================================================================================================================================================

# update dataset retraining information frame

update_dataset_retraining_information_title_label = ctk.CTkLabel(master=update_dataset_retraining_information_frame, text="Retraining Dataset", font=('Roboto', 25))
update_dataset_retraining_information_title_label.grid(row=1, column=1, columnspan=2, sticky='n')

list_update_dataset_retraining_information_label = ctk.CTkLabel(master=update_dataset_retraining_information_frame, text="List of Words:", font=('Roboto', 14))
list_update_dataset_retraining_information_label.grid(row=2, column=1, sticky='w', padx=5)

list_update_dataset_retraining_information_frame = ctk.CTkScrollableFrame(master=update_dataset_retraining_information_frame, corner_radius=10, orientation='vertical', width=145)
list_update_dataset_retraining_information_frame.grid(row=3, column=1, rowspan=2, sticky='nsew', padx=5, pady=5)

choose_word_update_dataset_retraining_information_label = ctk.CTkLabel(master=update_dataset_retraining_information_frame, text="Choose a Word:", font=('Roboto', 14))
choose_word_update_dataset_retraining_information_label.grid(row=2, column=2, sticky='w', padx=5)

choose_word_update_dataset_retraining_information_entry = ctk.CTkEntry(master=update_dataset_retraining_information_frame, corner_radius=10, font=("Roboto", 15))
choose_word_update_dataset_retraining_information_entry.grid(row=3, column=2, sticky='nsew', padx=5, pady=5)

notification_update_dataset_retraining_information_label = ctk.CTkLabel(master=update_dataset_retraining_information_frame, text=notification_update_dataset_retraining_information_0, font=('Roboto', 14), wraplength=200, anchor='center')
notification_update_dataset_retraining_information_label.grid(row=4, column=2, sticky='nsew', padx=5)

back_update_dataset_retraining_information_button = ctk.CTkButton(master=update_dataset_retraining_information_frame, text='Back', font=('Roboto', 14), corner_radius=10, command=lambda: to_update_dataset_frame(update_dataset_retraining_information_frame, update_dataset_frame))
back_update_dataset_retraining_information_button.grid(row=5, column=1, sticky='nsew', padx=5, pady=5)

next_update_dataset_retraining_information_button = ctk.CTkButton(master=update_dataset_retraining_information_frame, text='Next', font=('Roboto', 14), corner_radius=10, command=lambda: to_update_dataset_retraining_holistic_frame(update_dataset_retraining_information_frame, update_dataset_retraining_holistic_frame))
next_update_dataset_retraining_information_button.grid(row=5, column=2, sticky='nsew', padx=5, pady=5)

all_buttons.extend([back_update_dataset_retraining_information_button, next_update_dataset_retraining_information_button])

# =======================================================================================================================================================

# update dataset retraining holistic frame

update_dataset_retraining_holistic_title_label = ctk.CTkLabel(master=update_dataset_retraining_holistic_frame, text="Retraining Dataset", font=('Roboto', 25))
update_dataset_retraining_holistic_title_label.grid(row=1, column=1, columnspan=3, sticky='n')

update_dataset_retraining_holistic_camera_label = tk.Label(master=update_dataset_retraining_holistic_frame, font=('Roboto', 25))
update_dataset_retraining_holistic_camera_label.grid(row=2, column=1, columnspan=2, rowspan=3, sticky='nsew', padx=5, pady=5)

update_dataset_retraining_holistic_word_label = ctk.CTkLabel(master=update_dataset_retraining_holistic_frame, text='Word: Word', font=('Roboto', 14), anchor='w')
update_dataset_retraining_holistic_word_label.grid(row=2, column=3, sticky='nsew', padx=5, pady=5)

instructions_update_dataset_retraining_holistic = '''Instructions:

1. Click the start button to get started!

2. Only one person in front of the camera!

3. Do sign language!
'''

update_dataset_retraining_instruction_label = ctk.CTkLabel(master=update_dataset_retraining_holistic_frame, text=instructions_update_dataset_retraining_holistic, font=('Roboto', 14), anchor='nw', justify='left', wraplength=150)
update_dataset_retraining_instruction_label.grid(row=3, column=3, sticky='nsew', padx=5, pady=5)

update_dataset_retraining_holistic_notification_label = ctk.CTkLabel(master=update_dataset_retraining_holistic_frame, text=notification_update_dataset_retraining_holistic_0, font=('Roboto', 14), anchor='nw', justify='left', wraplength=150)
update_dataset_retraining_holistic_notification_label.grid(row=4, column=3, sticky='nsew', padx=5, pady=5)

start_update_dataset_retraining_holistic_button = ctk.CTkButton(master=update_dataset_retraining_holistic_frame, text='Start', font=('Roboto', 14), corner_radius=10, command=lambda: update_dataset_retraining_holistic_start_button_click())
start_update_dataset_retraining_holistic_button.grid(row=5, column=1, sticky='nsew', padx=5, pady=5)

back_update_dataset_retraining_holistic_button = ctk.CTkButton(master=update_dataset_retraining_holistic_frame, text='Back', font=('Roboto', 14), corner_radius=10, command=lambda: to_update_dataset_retraining_information_frame(update_dataset_retraining_holistic_frame, update_dataset_retraining_information_frame))
back_update_dataset_retraining_holistic_button.grid(row=6, column=1, sticky='nsew', padx=5, pady=5)

test_update_dataset_retraining_holistic_button = ctk.CTkButton(master=update_dataset_retraining_holistic_frame, text='Testing', font=('Roboto', 14), corner_radius=10, command=lambda: update_dataset_retraining_holistic_test_button_click())
test_update_dataset_retraining_holistic_button.grid(row=6, column=3, sticky='nsew', padx=5, pady=5)

bone_update_dataset_retraining_holistic_switch = ctk.CTkSwitch(master=update_dataset_retraining_holistic_frame, text='Bone', variable=ctk.StringVar(value='off'), onvalue='on', offvalue='off', bg_color='transparent', font=('Roboto', 14))
bone_update_dataset_retraining_holistic_switch.grid(row=5, column=3, sticky='w', padx=5, pady=5)

home_update_dataset_retraining_holistic_button = ctk.CTkButton(master=update_dataset_retraining_holistic_frame, text='Home', font=('Roboto', 14), corner_radius=10, command=lambda: to_home_frame(update_dataset_retraining_holistic_frame, home_frame))
home_update_dataset_retraining_holistic_button.grid(row=5, column=1, sticky='nsew', padx=5, pady=5)

update_dataset_retraining_holistic_progress_bar = ctk.CTkProgressBar(master=update_dataset_retraining_holistic_frame, orientation='horizontal')
update_dataset_retraining_holistic_progress_bar.grid(row=5, column=2, sticky='ew', padx=5, pady=5)
update_dataset_retraining_holistic_progress_bar.set(0)

update_dataset_retraining_holistic_progress_label = ctk.CTkLabel(master=update_dataset_retraining_holistic_frame, text='0%', font=('Roboto', 14))
update_dataset_retraining_holistic_progress_label.grid(row=6, column=2, sticky='nsew', padx=5, pady=5)

all_buttons.extend([start_update_dataset_retraining_holistic_button, back_update_dataset_retraining_holistic_button, test_update_dataset_retraining_holistic_button, home_update_dataset_retraining_holistic_button])
all_switches.append(bone_update_dataset_retraining_holistic_switch)
all_progress_bar.append(update_dataset_retraining_holistic_progress_bar)

# =======================================================================================================================================================

# delete dataset frame

delete_dataset_title_label = ctk.CTkLabel(master=delete_dataset_frame, text="Delete Dataset", font=('Roboto', 25))
delete_dataset_title_label.grid(row=1, column=1, columnspan=2, sticky='n')

list_delete_dataset_label = ctk.CTkLabel(master=delete_dataset_frame, text="List of Words:", font=('Roboto', 14))
list_delete_dataset_label.grid(row=2, column=1, sticky='w', padx=5)

list_delete_dataset_frame = ctk.CTkScrollableFrame(master=delete_dataset_frame, corner_radius=10, orientation='vertical', width=145)
list_delete_dataset_frame.grid(row=3, column=1, rowspan=2, sticky='nsew', padx=5, pady=5)

choose_word_delete_dataset_label = ctk.CTkLabel(master=delete_dataset_frame, text="Choose a Word:", font=('Roboto', 14))
choose_word_delete_dataset_label.grid(row=2, column=2, sticky='w', padx=5)

choose_word_delete_dataset_entry = ctk.CTkEntry(master=delete_dataset_frame, corner_radius=10, font=("Roboto", 15))
choose_word_delete_dataset_entry.grid(row=3, column=2, sticky='nsew', padx=5, pady=5)

notification_delete_dataset_label = ctk.CTkLabel(master=delete_dataset_frame, text=notification_delete_dataset_0, font=('Roboto', 14), wraplength=200, anchor='center')
notification_delete_dataset_label.grid(row=4, column=2, sticky='nsew', padx=5)

back_delete_dataset_button = ctk.CTkButton(master=delete_dataset_frame, text='Back', font=('Roboto', 14), corner_radius=10, command=lambda: to_dataset_frame(delete_dataset_frame, dataset_frame))
back_delete_dataset_button.grid(row=5, column=1, sticky='nsew', padx=5, pady=5)

agree_delete_dataset_button = ctk.CTkButton(master=delete_dataset_frame, text='Delete', font=('Roboto', 14), corner_radius=10, command=lambda: delete_dataset_button_click())
agree_delete_dataset_button.grid(row=5, column=2, sticky='nsew', padx=5, pady=5)

all_buttons.extend([back_delete_dataset_button, agree_delete_dataset_button])

# =======================================================================================================================================================

# translate frame

translate_title_label = ctk.CTkLabel(master=translate_frame, text="Translate", font=('Roboto', 25))
translate_title_label.grid(row=1, column=1, columnspan=4, sticky='n')

translate_camera_label = tk.Label(master=translate_frame, font=('Roboto', 25))
translate_camera_label.grid(row=2, column=1, rowspan=2, columnspan=2, sticky='nsew', padx=5, pady=5)

list_translate_label = ctk.CTkLabel(master=translate_frame, text='Translation List:', font=('Roboto', 14))
list_translate_label.grid(row=2, column=3, columnspan=2, sticky='w', padx=5, pady=5)

list_translate_frame = ctk.CTkScrollableFrame(master=translate_frame, corner_radius=10, orientation='vertical', width=200)
list_translate_frame.grid(row=3, column=3, columnspan=2, sticky='nsew', padx=5, pady=5)

start_translate_button = ctk.CTkButton(master=translate_frame, text='Start', font=('Roboto', 14), corner_radius=10, command=lambda: translate_start_button_click())
start_translate_button.grid(row=4, column=1, sticky='nsew', padx=5, pady=5)

back_translate_button = ctk.CTkButton(master=translate_frame, text='Back', font=('Roboto', 14), corner_radius=10, command=lambda: to_home_frame(translate_frame, home_frame))
back_translate_button.grid(row=5, column=1, sticky='nsew', padx=5, pady=5)

stop_translate_button = ctk.CTkButton(master=translate_frame, text='Stop', font=('Roboto', 14), corner_radius=10, command=lambda: translate_stop_button_click())
stop_translate_button.grid(row=5, column=4, sticky='nsew', padx=5, pady=5)

bone_translate_switch = ctk.CTkSwitch(master=translate_frame, text='Bone', variable=ctk.StringVar(value='off'), onvalue='on', offvalue='off', bg_color='transparent', font=('Roboto', 14))
bone_translate_switch.grid(row=4, column=4, sticky='w', padx=5, pady=5)

translate_label = ctk.CTkLabel(master=translate_frame, corner_radius=10, font=("Roboto", 15), justify='center', anchor='center')
translate_label.grid(row=4, column=2, columnspan=2, sticky='ew', padx=5, pady=5)

translate_progress_bar = ctk.CTkProgressBar(master=translate_frame, orientation='horizontal')
translate_progress_bar.grid(row=5, column=2, columnspan=2, sticky='ew', padx=5, pady=5)
translate_progress_bar.set(0)

all_buttons.extend([start_translate_button, back_translate_button, stop_translate_button])
all_switches.append(bone_translate_switch)
all_progress_bar.append(translate_progress_bar)

# =======================================================================================================================================================

# history frame

history_title_label = ctk.CTkLabel(master=history_frame, text="History", font=('Roboto', 25))
history_title_label.grid(row=1, column=1, columnspan=4, sticky='n')

list_history_label = ctk.CTkLabel(master=history_frame, text="History List:", font=('Roboto', 14))
list_history_label.grid(row=2, column=1, columnspan=2, sticky='w', padx=5)

list_history_frame = ctk.CTkScrollableFrame(master=history_frame, corner_radius=10, orientation='vertical', width=145)
list_history_frame.grid(row=3, column=1, columnspan=2, sticky='nsew', padx=5, pady=5)

list_translation_label = ctk.CTkLabel(master=history_frame, text="Translation List:", font=('Roboto', 14))
list_translation_label.grid(row=2, column=3, columnspan=2, sticky='w', padx=5)

list_translation_frame = ctk.CTkScrollableFrame(master=history_frame, corner_radius=10, orientation='vertical', width=145)
list_translation_frame.grid(row=3, column=3, columnspan=2, sticky='nsew', padx=5, pady=5)

back_history_button = ctk.CTkButton(master=history_frame, text='Back', font=('Roboto', 14), corner_radius=10, command=lambda: to_home_frame(history_frame, home_frame))
back_history_button.grid(row=4, column=1, sticky='nsew', padx=5, pady=5)

delete_history_button = ctk.CTkButton(master=history_frame, text='Delete', font=('Roboto', 14), corner_radius=10, command=lambda: history_delete_button_click())
delete_history_button.grid(row=4, column=4, sticky='nsew', padx=5, pady=5)

all_buttons.extend([back_history_button, delete_history_button])

# =======================================================================================================================================================

# tutorial frame

tutorial_title_label = ctk.CTkLabel(master=tutorial_frame, text="Tutorial", font=('Roboto', 25))
tutorial_title_label.grid(row=1, column=1, sticky='n')

tutorial_content_frame = ctk.CTkScrollableFrame(master=tutorial_frame, corner_radius=10, orientation='vertical')
tutorial_content_frame.grid(row=2, column=1, sticky='nsew', padx=5, pady=5)

# ----

tutorial_home_text = '''
Home Page:

On the home page, you can select various available features, namely dataset, history, translate, settings, and the tutorial you are currently reading.

'''
tutorial_home_label = ctk.CTkLabel(master=tutorial_content_frame, text=tutorial_home_text, font=('Roboto', 14), wraplength=550, anchor='w', justify='left')
tutorial_home_label.pack(side='top')

home_image = Image.open('D:/Project/InSign/assets/home_page.png')
home_image = home_image.resize((400, 255), Image.LANCZOS)
home_image_tk = ImageTk.PhotoImage(home_image)
image_tutorial_home_label = tk.Label(master=tutorial_content_frame, image=home_image_tk, borderwidth=0)
image_tutorial_home_label.pack(side='top', pady=10)

# ----

tutorial_dataset_text = '''
Dataset Page:

On the dataset page, you can manage your dataset with various features, namely adding, updating and deleting dataset.

'''
tutorial_dataset_label = ctk.CTkLabel(master=tutorial_content_frame, text=tutorial_dataset_text, font=('Roboto', 14), wraplength=550, anchor='w', justify='left')
tutorial_dataset_label.pack(side='top')

dataset_image = Image.open('D:/Project/InSign/assets/dataset_page.png')
dataset_image = dataset_image.resize((270, 290), Image.LANCZOS)
dataset_image_tk = ImageTk.PhotoImage(dataset_image)
image_tutorial_dataset_label = tk.Label(master=tutorial_content_frame, image=dataset_image_tk, borderwidth=0)
image_tutorial_dataset_label.pack(side='top', pady=10)

# ----

tutorial_add_dataset_text = '''
Add Dataset Page:

On the add dataset page, you can add word names and gestures. in gesture capture, you need to add different poses to enrich the dataset.

'''
tutorial_add_dataset_label = ctk.CTkLabel(master=tutorial_content_frame, text=tutorial_add_dataset_text, font=('Roboto', 14), wraplength=550, anchor='w', justify='left')
tutorial_add_dataset_label.pack(side='top')

add_dataset_image = Image.open('D:/Project/InSign/assets/add_dataset_page.png')
add_dataset_image = add_dataset_image.resize((605, 490), Image.LANCZOS)
add_dataset_image_tk = ImageTk.PhotoImage(add_dataset_image)
image_tutorial_add_dataset_label = tk.Label(master=tutorial_content_frame, image=add_dataset_image_tk, borderwidth=0)
image_tutorial_add_dataset_label.pack(side='top', pady=10)

# ----

tutorial_update_dataset_text = '''
Update Dataset Page:

On the dataset update page, you can choose rewording or retraining. In rewording, you can change the words in the saved dataset, without changing the gestures. conversely, in retraining you can change the gestures on certain words, without changing the word.

'''

tutorial_update_dataset_label = ctk.CTkLabel(master=tutorial_content_frame, text=tutorial_update_dataset_text, font=('Roboto', 14), wraplength=550, anchor='w', justify='left')
tutorial_update_dataset_label.pack(side='top')

rewording_dataset_image = Image.open('D:/Project/InSign/assets/rewording_dataset_page.png')
rewording_dataset_image = rewording_dataset_image.resize((400, 320), Image.LANCZOS)
rewording_dataset_image_tk = ImageTk.PhotoImage(rewording_dataset_image)
image_tutorial_rewording_dataset_label = tk.Label(master=tutorial_content_frame, image=rewording_dataset_image_tk, borderwidth=0)
image_tutorial_rewording_dataset_label.pack(side='top', pady=10)

retraining_dataset_image = Image.open('D:/Project/InSign/assets/retraining_dataset_page.png')
retraining_dataset_image = retraining_dataset_image.resize((605, 490), Image.LANCZOS)
retraining_dataset_image_tk = ImageTk.PhotoImage(retraining_dataset_image)
image_tutorial_retraining_dataset_label = tk.Label(master=tutorial_content_frame, image=retraining_dataset_image_tk, borderwidth=0)
image_tutorial_retraining_dataset_label.pack(side='top', pady=10)

# ----

tutorial_delete_dataset_text = '''
Delete Dataset Page:

On the delete dataset page, you can delete words and gestures in the dataset and training data permanently. once data is deleted, it cannot be reused in subsequent features.

'''

tutorial_delete_dataset_label = ctk.CTkLabel(master=tutorial_content_frame, text=tutorial_delete_dataset_text, font=('Roboto', 14), wraplength=550, anchor='w', justify='left')
tutorial_delete_dataset_label.pack(side='top')

delete_dataset_image = Image.open('D:/Project/InSign/assets/delete_dataset_page.png')
delete_dataset_image = delete_dataset_image.resize((400, 290), Image.LANCZOS)
delete_dataset_image_tk = ImageTk.PhotoImage(delete_dataset_image)
image_tutorial_delete_dataset_label = tk.Label(master=tutorial_content_frame, image=delete_dataset_image_tk, borderwidth=0)
image_tutorial_delete_dataset_label.pack(side='top', pady=10)

# ----

tutorial_history_text = '''
History Page:

On the history page, you can see all the history of translations that have been made using the translate feature. This history has different IDs which are arranged based on the date and time the translation started. and the translation results have date and time information for each sentence. You can review the translation results, or delete them.

'''

tutorial_history_label = ctk.CTkLabel(master=tutorial_content_frame, text=tutorial_history_text, font=('Roboto', 14), wraplength=550, anchor='w', justify='left')
tutorial_history_label.pack(side='top')

history_image = Image.open('D:/Project/InSign/assets/history_page.png')
history_image = history_image.resize((540, 420), Image.LANCZOS)
history_image_tk = ImageTk.PhotoImage(history_image)
image_tutorial_history_label = tk.Label(master=tutorial_content_frame, image=history_image_tk, borderwidth=0)
image_tutorial_history_label.pack(side='top', pady=10)

# ----

tutorial_translate_text = '''
Translate Page:

On the translate page, you can translate words using gestures. each resulting word can be arranged into the desired sentence, and moved to the conversation box. The conversation box will contain the entire conversation along with the date and time for each sentence. and if the translation process is stopped by pressing the stop button, all the conversation results in the box will be saved and moved to the history feature.

'''

tutorial_translate_label = ctk.CTkLabel(master=tutorial_content_frame, text=tutorial_translate_text, font=('Roboto', 14), wraplength=550, anchor='w', justify='left')
tutorial_translate_label.pack(side='top')

translate_image = Image.open('D:/Project/InSign/assets/translate_page.png')
translate_image = translate_image.resize((715, 495), Image.LANCZOS)
translate_image_tk = ImageTk.PhotoImage(translate_image)
image_tutorial_translate_label = tk.Label(master=tutorial_content_frame, image=translate_image_tk, borderwidth=0)
image_tutorial_translate_label.pack(side='top', pady=10)

# ----

tutorial_setting_text = '''
Setting Page:

On the settings page, you can set the appearance mode in the form of system (follows the current PC color mode), light and dark. Apart from that, you can change the application theme, which will change the color of various widgets, such as buttons, progress bars and switches. Then, you can set the bones that can be activated in the translate and retraining features. You can set to display which parts of the body landmarks or rectangles will appear. then you can see the training results in a new window when you press the open button. in this feature, the training data will be updated automatically, and the level of accuracy for each word will be shown. then you can separate the settings from the main window by pressing the pop out button, so you can change the settings while you are using other features.

'''

tutorial_setting_label = ctk.CTkLabel(master=tutorial_content_frame, text=tutorial_setting_text, font=('Roboto', 14), wraplength=550, anchor='w', justify='left')
tutorial_setting_label.pack(side='top')

setting_image = Image.open('D:/Project/InSign/assets/setting_page.png')
setting_image = setting_image.resize((300, 510), Image.LANCZOS)
setting_image_tk = ImageTk.PhotoImage(setting_image)
image_tutorial_setting_label = tk.Label(master=tutorial_content_frame, image=setting_image_tk, borderwidth=0)
image_tutorial_setting_label.pack(side='top', pady=10)

training_result_image = Image.open('D:/Project/InSign/assets/training_result_page.png')
training_result_image = training_result_image.resize((300, 420), Image.LANCZOS)
training_result_image_tk = ImageTk.PhotoImage(training_result_image)
image_tutorial_training_result_label = tk.Label(master=tutorial_content_frame, image=training_result_image_tk, borderwidth=0)
image_tutorial_training_result_label.pack(side='top', pady=10)

# ----

tutorial_tutorial_text = '''
Tutorial Page:

The tutorial page you are currently reading will provide information about how to use this application.

'''

tutorial_tutorial_label = ctk.CTkLabel(master=tutorial_content_frame, text=tutorial_tutorial_text, font=('Roboto', 14), wraplength=550, anchor='w', justify='left')
tutorial_tutorial_label.pack(side='top')

tutorial_image = Image.open('D:/Project/InSign/assets/tutorial_page.png')
tutorial_image = tutorial_image.resize((470, 420), Image.LANCZOS)
tutorial_image_tk = ImageTk.PhotoImage(tutorial_image)
image_tutorial_tutorial_label = tk.Label(master=tutorial_content_frame, image=tutorial_image_tk, borderwidth=0)
image_tutorial_tutorial_label.pack(side='top', pady=10)

# ----

tutorial_email_text = '''
Of course, this application is still under further research. However, if you have criticism or suggestions that can help the development of this application, feel free to send your criticism and suggestions to the following email:

adyamp7000@gmail.com

Enjoy:)

'''

tutorial_email_label = ctk.CTkLabel(master=tutorial_content_frame, text=tutorial_email_text, font=('Roboto', 14), wraplength=550, anchor='w', justify='left')
tutorial_email_label.pack(side='top')

back_tutorial_button = ctk.CTkButton(master=tutorial_frame, text='Back', font=('Roboto', 14), corner_radius=10, command=lambda: to_home_frame(tutorial_frame, home_frame), width=200)
back_tutorial_button.grid(row=3, column=1, sticky='nsw', padx=5, pady=5)

all_buttons.append(back_tutorial_button)

# =======================================================================================================================================================

add_dataset_information_frame.tkraise(add_dataset_holistic_frame)
delete_dataset_frame.tkraise(add_dataset_information_frame)
update_dataset_retraining_holistic_frame.tkraise(delete_dataset_frame)
update_dataset_retraining_information_frame.tkraise(update_dataset_retraining_holistic_frame)
update_dataset_rewording_frame.tkraise(update_dataset_retraining_information_frame)
update_dataset_frame.tkraise(update_dataset_rewording_frame)

tutorial_frame.tkraise(update_dataset_frame)
setting_frame.tkraise(tutorial_frame)
translate_frame.tkraise(setting_frame)
history_frame.tkraise(translate_frame)
dataset_frame.tkraise(history_frame)
home_frame.tkraise(dataset_frame)

app.mainloop()
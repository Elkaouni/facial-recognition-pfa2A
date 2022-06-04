import tkinter as tk
import cv2
from tkinter import *
from tkinter import ttk 
from PIL import Image, ImageTk
from random import randint
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



print('hello')


model = model_from_json(open("model1/model.json", "r").read())
model.load_weights('model1/model.h5') 
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')    


def openLiveStream():
    root = Toplevel(window)
    root.geometry("1180x650")
    root.configure(bg="#d4f6ff")
    
    fr1 = tk.Frame(master=root, width=600, height=50,bg="#d4f6ff")
    fr1.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

    exit_button = Button(root, text="Exit", font=("Courrier", 25),bg='#0F094A', fg='#ffffff', command=root.destroy)
    exit_button.pack(pady=20)

    f1 = LabelFrame(fr1, bg="white")
    f1.grid(row=3, column=5,padx=(30,0), pady=(20,0))
    
    
    f2 = LabelFrame(fr1, bg="#0F094A")
    f2.grid(row=3, column=1, padx=(20,0), pady=(20,0))
    
    Label(f2, text="Predicted Expressions (%)", font=("times new roman", 20, "bold"), bg="#d4f6ff", fg="#0F094A").grid(pady=(0,30))
    
    Label(f2, text="Happy", font=("times new roman", 15, "bold"), bg="#0F094A", fg="white").grid(sticky = W)
    Pb1 = ttk.Progressbar(f2, orient='horizontal', mode='determinate', length=500)
    Pb1.grid()
    
    Label(f2, text="Angry", font=("times new roman", 15, "bold"), bg="#0F094A", fg="white").grid(pady=(16,0), sticky = W)
    Pb2 = ttk.Progressbar(f2, orient='horizontal', mode='determinate', length=500)
    Pb2.grid()
    
    Label(f2, text="Disgusted", font=("times new roman", 15, "bold"), bg="#0F094A", fg="white").grid(pady=(16,0), sticky = W)
    Pb3 = ttk.Progressbar(f2, orient='horizontal', mode='determinate', length=500)
    Pb3.grid()
    
    Label(f2, text="Fear", font=("times new roman", 15, "bold"), bg="#0F094A", fg="white").grid(pady=(16,0), sticky = W)
    Pb4 = ttk.Progressbar(f2, orient='horizontal', mode='determinate', length=500)
    Pb4.grid()
    
    Label(f2, text="Sad", font=("times new roman", 15, "bold"), bg="#0F094A", fg="white").grid(pady=(16,0), sticky = W)
    Pb5 = ttk.Progressbar(f2, orient='horizontal', mode='determinate', length=500)
    Pb5.grid()
    
    Label(f2, text="Surprised", font=("times new roman", 15, "bold"), bg="#0F094A", fg="white").grid(pady=(16,0), sticky = W)
    Pb6 = ttk.Progressbar(f2, orient='horizontal', mode='determinate', length=500)
    Pb6.grid()
    
    Label(f2, text="Neutral", font=("times new roman", 15, "bold"), bg="#0F094A", fg="white").grid(pady=(16,0), sticky = W)
    Pb7 = ttk.Progressbar(f2, orient='horizontal', mode='determinate', length=500)
    Pb7.grid()
    
    Label(f1, text="Live Stream", font=("times new roman", 20, "bold"), bg="white", fg="#0F094A").grid()
    L1 = Label(f1, bg="white")
    L1.grid()
    
    
    cap = cv2.VideoCapture(0)
    #If the camera was not opened sucessfully
    if not cap.isOpened():  
        print("Cannot open camera")
        exit()
        
        
    angry_rate = 0
    disgusted_rate = 0 
    fear_rate = 0 
    happy_rate = 0 
    sad_rate = 0 
    surprise_rate = 0 
    neutral_rate = 0
            
    while True:
        #read frame by frame and get return whether there is a stream or not
        ret, img=cap.read()

        #If no frames recieved, then break from the loop
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        #Change the frame to greyscale  
        gray_image= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_image,1.32,5)

        #Draw Triangles around the faces detected
        for (x,y,w,h) in faces_detected:
            cv2.rectangle(img,(x,y), (x+w,y+h), (255,0,0), thickness=7)
            roi_gray=gray_image[y:y+w,x:x+h]
            roi_gray=cv2.resize(roi_gray,(48,48))

            #Processes the image and adjust it to pass it to the model
            image_pixels = tf.keras.preprocessing.image.img_to_array(roi_gray)
            #plt.imshow(image_pixels)
            #plt.show()
            image_pixels = np.expand_dims(image_pixels, axis = 0)
            image_pixels /= 255

            #Get the prediction of the model
            #predictions = model.predict(image_pixels)
            predictions = model.predict(image_pixels)
            #print(predictions)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion_prediction = emotion_detection[max_index]

            #print('Prediction: Right Now you are: ' + emotion_prediction)

            angry_rate = np.take(predictions[0],0)*100
            disgusted_rate = np.take(predictions[0],1)*100
            fear_rate = np.take(predictions[0],2)*100
            happy_rate = np.take(predictions[0],3)*100
            sad_rate = np.take(predictions[0],4)*100
            surprise_rate = np.take(predictions[0],5)*100
            neutral_rate = np.take(predictions[0],6)*100

            #print('---- Angry: '+str(angry_rate)+'%')
            #print('---- Disgust: '+str(disgusted_rate)+'%')
            #print('---- Fear: '+str(fear_rate)+'%')
            #print('---- Happy: '+str(happy_rate)+'%')
            #print('---- Sad: '+str(sad_rate)+'%')
            #print('---- Surprise: '+str(surprise_rate)+'%')
            #print('---- Neutral: '+str(neutral_rate)+'%')

            #Write on the frame the emotion detected
            cv2.putText(img,emotion_prediction,(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        
        
        #img = cap.read()[1]
        img = cv2.resize(img, (600,450))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        L1['image'] = img
        Pb1['value'] = happy_rate   #happy
        Pb2['value'] = angry_rate   #angry
        Pb3['value'] = disgusted_rate   #disgusted
        Pb4['value'] = fear_rate  #scared
        Pb5['value'] = sad_rate   #sad
        Pb6['value'] = surprise_rate   #surprised
        Pb7['value'] = neutral_rate   #neutral
        root.update()

   




window = tk.Tk()
window.title("FeelMeApp")
window.config(background="#d4f6ff")
window.geometry("1080x650")
window.resizable(width=False, height=False)

label_title = Label(window, text="Welcome to Feel Me App", font=("Helvetica", 30), bg='#d4f6ff', fg='black')
label_title.pack()

frame1 = tk.Frame(master=window, width=50, height=100, bg="#d4f6ff")
frame1.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

image = Image.open(".\\images\\logo_emotions.png")
photo = ImageTk.PhotoImage(image.resize((400, 400), Image.ANTIALIAS))
label = Label(frame1, image=photo, bg='#d4f6ff')
label.image = photo
label.pack()
label.place(anchor='center', relx=0.5, rely=0.5)

        
        
frame2 = tk.Frame(master=window, width=600, bg="#d4f6ff")
frame2.pack(fill=tk.Y, side=tk.RIGHT)
#frame2.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)


frame21 = tk.Frame(master=frame2, width=600, height=200, bg="#d4f6ff")
#frame21.pack(fill=tk.X)
frame21.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

text = Label(frame21, text="Feel Me App is a system based on Facial Emotion \nRecognition technology that analyses facial expressions from\n"
             +"videos in order to reveal information on your emotional state.\nThe system automatically analyzes the expressions happy, \nsad,angry, "
             +"surprised, scared, disgusted, and neutral.", font=("Helvetica", 15), bg='#d4f6ff', fg='black') 
text.pack()
text.place(anchor='center', relx=0.5, rely=0.5)


frame22 = tk.Frame(master=frame2, width=600, height=50, bg="#d4f6ff")
#frame22.pack(fill=tk.X)
frame22.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=True)

y_button = Button(frame22, text="Start now", font=("Courrier", 25), bg='#0F094A', fg='#ffffff', command = openLiveStream)
y_button.pack(pady=25)


window.mainloop()

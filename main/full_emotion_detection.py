import mysql.connector

import tkinter as tk
from tkinter import *
from tkinter import messagebox, ttk 
from PIL import Image, ImageTk

import cv2
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing import image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import os



full_path = "C:\\Users\\wichy\\Desktop\\ENSIAS 2A\\PFA-reconnaissance-faciale\\Face_recognizer"
classifer_path= "classifier.xml"
faceCascade_path= "haarcascade_frontalface_default.xml"
user_pfp_path= "C:\\Users\\wichy\\Desktop\\ENSIAS 2A\\PFA-reconnaissance-faciale\\images\\user.png"

data_max_img=50
db_user="root"
db_pwd="root"
db_name="feel_me"


# we load the trained model saved in the xml file
def load_model():
    model = model_from_json(open("model1/model.json", "r").read())
    model.load_weights('model1/model.h5') 
    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')    
    return model


#to disable X button
def disable_event():
   pass


def show_quote(i): #('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    mydb=mysql.connector.connect(
                host="localhost",
                user=db_user,
                passwd=db_pwd,
                database=db_name
            )
       
    if i==0 :
        #quote = "You're feeling angry?"
        emo="You seem pretty angry. Remember, "
        emotion = "where anger=1"
    elif i==1 :
        #quote = "You're feeling disgusted?"
        emo="Feelin disgusted? Remember, "
        emotion = "where disgust=1"
    elif i==2 :
        #quote = "You're feeling fear?"
        emo="Is something worrying you i sense fear... Don't worry, "
        emotion = "where fear=1"
    elif i==3 :
        #quote = "You're feeling happy?"
        emo="You seem quite happy! Remember, "
        emotion = "where happiness=1=1=1"
    elif i==4 :
        #quote = "You're feeling sad?"
        emo="Please don't be sad. "
        emotion = "where sadness=1=1"
    elif i==5 :
        #quote = "You're feeling surprise?"
        emo="You were surprised today. "
        emotion = "where surprise=1"
    else :
        #quote = "You're feeling neutral?"
        emo="Random quote of today:\n"
        emotion = ""
    
    mycursor=mydb.cursor()
    sql = """select quote from quotes """+emotion+"""  order by rand() limit 1;"""
    mycursor.execute(sql)
    quote = mycursor.fetchone()
    quote = emo + str(tuple(quote))
    return quote



def openLiveStream(window):
    global happiness
    global neutrality
    global fear
    global sadness
    global anger
    global disgust
    global surprise
    
    happiness=0
    neutrality=0
    fear=0
    sadness=0
    anger=0
    surprise=0
    disgust=0
    
    window.destroy()
    root = tk.Tk()
    #root = Toplevel(window)
    root.title("FeelMeApp - Live stream")
    root.geometry("1180x650")
    root.configure(bg="#d4f6ff")   
    
    model = model_from_json(open("model1/model.json", "r").read())
    model.load_weights('model1/model.h5') 
    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')    


    def openMenu():
        root.destroy()
        global window 
        window= tk.Tk()
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

        frame21 = tk.Frame(master=frame2, width=600, height=150, bg="#d4f6ff")
        frame21.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

        text = Label(frame21, text="Feel Me App is a system based on Facial Emotion \nRecognition technology that analyses facial expressions from\n"
                     +"videos in order to reveal information on your emotional state.\nThe system automatically analyzes the expressions happy, \nsad,angry, "
                     +"surprised, scared, disgusted, and neutral.", font=("Helvetica", 15), bg='#d4f6ff', fg='black') 
        text.pack()
        text.place(anchor='n', relx=0.5, rely=0.5)


        frame22 = tk.Frame(master=frame2, width=600, height=30, bg="#d4f6ff")
        frame22.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=True)

        y_button = Button(frame22, text="Start now", font=("Courrier", 25), bg='#0F094A', fg='#ffffff', command = openLiveStream)
        y_button.pack(pady=25)

        frame23 = tk.Frame(master=frame2, width=600, height=10, bg="#d4f6ff")
        frame23.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

        #('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        #print('---- Angry: '+str(anger)+'%')
        #print('---- Disgust: '+str(disgust)+'%')
        #print('---- Fear: '+str(fear)+'%')
        #print('---- Happy: '+str(happiness)+'%')
        #print('---- Sad: '+str(sadness)+'%')
        #print('---- Surprise: '+str(surprise)+'%')
        #print('---- Neutral: '+str(neutrality)+'%')
            
        list_feelings = [ anger, disgust, fear, happiness, sadness, surprise, neutrality]
        max_feel = max(list_feelings)
        max_index = list_feelings.index(max_feel)
        quote = show_quote(max_index)
        print("quote is -------------- "+ quote)
        quoting = Label(frame23, text=quote, font=("Helvetica", 15, "italic"), bg='#d4f6ff', fg='#0F094A', wraplengt=500) 
        quoting.pack()
        quoting.place(anchor='center', relx=0.5, rely=0.5)

        window.mainloop()
        
        
    #Disable the Close Window Control Icon
    root.protocol("WM_DELETE_WINDOW", disable_event)
    
    fr1 = tk.Frame(master=root, width=600, height=50,bg="#d4f6ff")
    fr1.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

    exit_button = Button(root, text="Exit", font=("Courrier", 25),bg='#0F094A', fg='#ffffff', command=openMenu)
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
        
        happiness += happy_rate
        neutrality+= neutral_rate
        surprise+= surprise_rate
        fear+= fear_rate
        sadness+= sad_rate
        anger+= angry_rate
        disgust+= disgusted_rate
        
        root.update()




def generate_dataset():
    #if(e1.get()=="" or e2.get()==""):
        #messagebox.showinfo('Result', 'Please provide completed details of the user')
    #else:
    mydb=mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="root",
        database="feel_me"
        )
    mycursor=mydb.cursor()
    mycursor.execute("SELECT * from authorized_user")
    myresult=mycursor.fetchall()
    id=1
    for x in myresult:
        id+=1
        
    sql="insert into authorized_user(id,username,password) values(%s,%s,%s)"
    val=(id,e1.get(),e2.get())
    mycursor.execute(sql,val)
    mydb.commit()
        
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    def face_cropped(img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor = 1.3
        # minimum neighbor = 5

        if faces is ():
            return None
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h,x:x+w]
        return cropped_face
    cap = cv2.VideoCapture(0)
    img_id = 0

    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id+=1
            face = cv2.resize(face_cropped(frame), (200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = full_path+ "\\data\\user."+str(id)+"."+str(img_id)+".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

            cv2.imshow("Cropped face", face)

        if cv2.waitKey(1)==13 or int(img_id)==data_max_img: #13 is the ASCII character of Enter
            break
    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo('Result', 'Generating dataset completed!!')
    
    
    
    
    
def train_classifier():
    data_dir = full_path + "\\data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    
    faces = []
    ids = []
    
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
        
        faces.append(imageNp)
        ids.append(id)
        
    ids = np.array(ids)
    
    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write(classifer_path)
    messagebox.showinfo('Result', 'Training dataset completed')
    
    
        

def detect_face():  
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
        name = "????"
        for (x,y,w,h) in features:
            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2 )

            id, pred = clf.predict(gray_img[y:y+h,x:x+w])
            confidence = int(100*(1-pred/300))
            
            mydb=mysql.connector.connect(
                host="localhost",
                user=db_user,
                passwd=db_pwd,
                database=db_name
            )
            mycursor=mydb.cursor()
            mycursor.execute("select username from authorized_user where id="+str(id))
            s = mycursor.fetchone()
            s = ''+''.join(s)
            if confidence>80:
                name = s
                cv2.putText(img, s, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                name = "UNKNOWN"
                cv2.putText(img, "UNKNOWN", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
        return img, name

    # loading classifier
    faceCascade = cv2.CascadeClassifier(faceCascade_path)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        img, username = draw_boundary(img, faceCascade, 1.3, 6, (255,255,255), "Face", clf)
        cv2.imshow("face Detection", img)

        if cv2.waitKey(1)==13:
            break
    video_capture.release()
    cv2.destroyAllWindows()
    return username







def create_menu( given_username, entry):
    txt = "Welcome "+given_username +"!"
   # Label(home, text=given_username, font=("Courrier", 40), bg='#A1C6E7', fg='white').pack()
   # home.mainloop()

    window = tk.Tk()
    window.title("FeelMeApp")
    window.config(background="#d4f6ff")
    window.geometry("1080x650")
    window.resizable(width=False, height=False)

    label_title = Label(window, text=txt, font=("Helvetica", 30), bg='#d4f6ff', fg='black')
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


    frame21 = tk.Frame(master=frame2, width=600, height=150, bg="#d4f6ff")
    frame21.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

    text = Label(frame21, text="Feel Me App is a system based on Facial Emotion \nRecognition technology that analyses facial expressions from\n"
                 +"videos in order to reveal information on your emotional state.\nThe system automatically analyzes the expressions happy, \nsad,angry, "
                 +"surprised, scared, disgusted, and neutral.", font=("Helvetica", 15), bg='#d4f6ff', fg='black') 
    text.pack()
    text.place(anchor='n', relx=0.5, rely=0.5)


    frame22 = tk.Frame(master=frame2, width=600, height=25, bg="#d4f6ff")
    frame22.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=True)

    y_button = Button(frame22, text="Start now", font=("Courrier", 25), bg='#0F094A', fg='#ffffff', command = lambda : openLiveStream(window))
    y_button.pack(pady=25)

    frame23 = tk.Frame(master=frame2, width=600, height=10, bg="#d4f6ff")
    frame23.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

    quote="How are you feeling today?"
    quoting = Label(frame23, text=quote, font=("Helvetica", 14, "italic"), bg='#d4f6ff', fg='black', wraplengt=600) 
    quoting.pack()
    quoting.place(anchor='center', relx=0.5, rely=0.5)


    window.mainloop()
        
        
def verify_password(given_username, entry):
    given_password=entry.get()
    if(given_password ==""):
        messagebox.showinfo('Result', 'Please provide a password')
    else:
        mydb=mysql.connector.connect(
                host="localhost",
                user=db_user,
                passwd=db_pwd,
                database=db_name
            )
        mycursor=mydb.cursor()
        sql = """SELECT password FROM `authorized_user` WHERE `username` = %s"""
        mycursor.execute(sql, (given_username,))
        password = mycursor.fetchone()
        password = ''+''.join(password)
        if(given_password != password):
            messagebox.showinfo('Result', 'Invalid password')
        else:
            login.destroy()
            #home = Tk()
            #home.title("Feel me APP")
            #home.geometry("500x350")
            #home.config(background="#A1C6E7")
            create_menu( given_username, entry)
            
        
        
        
        
        
def open_login(username):
    w.destroy()
    global login
    login = Tk()
    login.title("Feel me APP")
    login.geometry("350x500")
    login.resizable(0,0)
    login.config(background="#A1C6E7")

    Frame(login, width=250, height=400, bg = 'white').place(x=50, y=50)

    l3 = Label(login, text="Username", font=("consolas", 13), bg='white')
    l3.place(x=80, y=190)

    e3=Entry(login, width=20, border=0, font=1)
    e3.insert(END, username)
    e3.place(x=80, y=220)

    l4 = Label(login, text="Password", font=("consolas", 13), bg='white')
    l4.place(x=80, y=270)

    e4=Entry(login, width=20, border=0, font=1, show="*")
    e4.place(x=80, y=300)

    Frame(login, width=180, height=2, bg='#141414').place(x=80, y=240)
    Frame(login, width=180, height=2, bg='#141414').place(x=80, y=320)
    
    #global img1
    #img2 = ImageTk.PhotoImage((Image.open("user.png")).resize((100, 100)))
    #Label(image = img, border=0, justify=CENTER).place(x=120, y=60)

    #Button(w, text="Detect the face", width=20, height=2, bg='#A1C6E7', fg='white', command=generate_dataset).place(x=100, y=375)
    #Button(w, text="Detect the face", width=20, height=2, bg='#A1C6E7', fg='white', command=train_classifier).place(x=100, y=375)
    Button(login, text="Login", width=20, height=2, bg='#A1C6E7', fg='white', command=lambda given_username=username, entry=e4 : verify_password(given_username, entry)).place(x=100, y=345)
    

    login.mainloop()
    
    
def new_user():
    if(e1.get()=="" or e2.get()==""):
        messagebox.showinfo('Result', 'Please provide completed details of the user')
    else:
        generate_dataset()
        train_classifier()
        username = detect_face()
        w.destroy()
        home = Tk()
        home.title("Feel me APP")
        home.geometry("500x350")
        home.config(background="#A1C6E7")
        Label(home, text=username, font=("Courrier", 40), bg='#A1C6E7', fg='white').pack()
        home.mainloop()
        
        
    
    
def have_already_account():
    username = detect_face()
    if(username=="UNKNOWN"):
        messagebox.showinfo('Result', 'You are not an authorized user, please register first')
    else:
        open_login(username)
    #if
    #if(e2.get()==""):
        #messagebox.showinfo('Result', 'Please provide password')
    #else:
                
 
    
w = Tk()
w.title("Feel me APP")
w.geometry("350x500")
w.resizable(0,0)
w.config(background="#A1C6E7")

Frame(w, width=250, height=400, bg = 'white').place(x=50, y=50)

l1 = Label(w, text="Username", font=("consolas", 13), bg='white')
l1.place(x=80, y=190)

e1=Entry(w, width=20, border=0, font=1)
e1.place(x=80, y=220)

l2 = Label(w, text="Password", font=("consolas", 13), bg='white')
l2.place(x=80, y=270)

e2=Entry(w, width=20, border=0, font=1, show="*")
e2.place(x=80, y=300)

Frame(w, width=180, height=2, bg='#141414').place(x=80, y=240)
Frame(w, width=180, height=2, bg='#141414').place(x=80, y=320)

img = ImageTk.PhotoImage((Image.open(user_pfp_path)).resize((100, 100)))

Label(image = img, border=0, justify=CENTER).place(x=120, y=60)

#Button(w, text="Detect the face", width=20, height=2, bg='#A1C6E7', fg='white', command=generate_dataset).place(x=100, y=375)
#Button(w, text="Detect the face", width=20, height=2, bg='#A1C6E7', fg='white', command=train_classifier).place(x=100, y=375)
Button(w, text="New user", width=20, height=2, bg='#A1C6E7', fg='white', command= new_user).place(x=100, y=345)
Button(w, text="Have already an account", width=20, height=2, bg='#A1C6E7', fg='white', command= have_already_account).place(x=100, y=400)

w.mainloop()
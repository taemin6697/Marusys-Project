# -*- coding: utf-8 -*-
import cv2
import numpy as np
from gtts import gTTS
import time
import speech_recognition as sr
import pygame
from model import getBaseModel_KNN
#from Button import PrepareButton, ButtonPushed
#from dashpy import dashReset

y = []
y_object = []

button = [13, 18, 31, 29, 36] # buttons 1, 2, 3, 4
#LED = [11, 15, 3]

global activations, activations_object

def playsound(file):

    pygame.mixer.init()
    pygame.mixer.music.load("saved_tts/{}.mp3".format(file))
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy() == True:
        continue

def get_audio():
    
    r = sr.Recognizer()
    r.energy_threshold = 1000

    with sr.Microphone() as source:
        print("wait...")
        time.sleep(1)
        print("recording...")
        audio = r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio, language="ko-KR")

        except:
            print("Error")
            return ""

    return said


def get_name():
    while True:
        playsound('ask_name')
        print("이름이 뭔가요?")

        name = get_audio()
        if name == "":
            playsound('ask_again')
            
            print("다시 한번 말씀해주세요...")
            continue
        else:
            tts = gTTS(text="{}가 맞나요".format(name), lang="ko", slow=False)
            tts.save("confirm")
            
            playsound('confirm')
            
            print(name)
            print("이 이름이 맞나요?")
            response = get_audio()

            if response == "" or response in ["아니오", "아니요", "아니", "아리", "아니요 아니요"]:
                print("아니요")
                continue

            else:
                print("네")
                break

    return name


def get_image(capture, i, IMG_SIZE, face_detector, name):
    
    
    global activations_previous
    global activation_previous
    global activations, activations_object

    IMG_SHAPE = IMG_SIZE + (3,)
    count = 0
    z = 0

    playsound('caputure_instruction')    
    base_model = getBaseModel_KNN(IMG_SHAPE)

    while True:
        _, frame = capture.read()
        faces = face_detector.detectMultiScale(frame, 1.03, 10, minSize=(150,150))
        arr = np.asarray(faces)

        for (a,b,w,h) in faces:
            cv2.rectangle(frame, (a,b), (a+w,b+h), (255,0,0), 5)

        frame_fliped = cv2.flip(frame, 1)
        cv2.imshow("videoFrame",frame)
        key = cv2.waitKey(200)

        if z == 10:
            break     
                                    

        if (len(faces) == 1):   
            playsound('capture')
                
            faceframe = cv2.resize(frame[b:b+h,a:a+w], IMG_SIZE)

            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            data[0] = faceframe
                
            activation = base_model(data, training=False)
            print(type(activation))
            activation = activation.numpy().reshape(1, -1)
                           

                            

            if i == 0:
                    
                if count == 0:
                    if 'activations' in globals():
                        activation_previous = activations
                        activations = np.concatenate((activation_previous,activation), axis=0)
                        activation_previous = activations
                        print(activations.shape)
                    else:
                        activation_previous = activation   

                elif count == 1:
                    activations = np.concatenate((activation_previous,activation), axis=0)
                    activations_previous = activations
                    print(activations.shape)                       
                        
                else:
                    activations = np.concatenate((activations_previous,activation), axis=0)
                    activations_previous = activations
                    print(activations.shape)
                        

            else:
                    
                if 'activations_previous' not in locals():
                    activations_previous = activations
                    
                activations = np.concatenate((activations_previous,activation), axis=0)
                activations_previous = activations
                print(activations.shape)
                    
        
            y.append(i)                
            print(y)

            count += 1
            z += 1

            time.sleep(0.5)
                
        if (len(faces) > 1):   
            playsound('oneatatime')
                
            print('{}님 이외의 사람들은 비켜주세요.'.format(name))
            time.sleep(0.5)

        

    return count, activations

def get_image_object(capture, i, IMG_SIZE, face_detector, name):

    
    
    
    global activations_previous_object
    global activation_previous_object
    global activations, activations_object

    
    
    IMG_SHAPE = IMG_SIZE + (3,)
    count = 0
    z = 0

    playsound('caputure_instruction_object')
    base_model = getBaseModel_KNN(IMG_SHAPE)

    while True:
        _, frame = capture.read()
        faces = face_detector.detectMultiScale(frame, 1.03, 10, minSize=(150,150))
        arr = np.asarray(faces)

        for (a,b,w,h) in faces:
            cv2.rectangle(frame, (a,b), (a+w,b+h), (255,0,0), 5)

        frame_fliped = cv2.flip(frame, 1)

        key = cv2.waitKey(200)

        if z == 10:
            break
        
        
        
                    

            
        # 사물인식
        if arr.shape == (0,):

            playsound('capture')
            frame = cv2.resize(frame, IMG_SIZE)                                

            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            data[0] = frame
                
            activation = base_model(data, training=False)
            activation = activation.numpy().reshape(1, -1)
                      
                      

            if i == 0:
                    
                if count == 0:
                    if 'activations_object' in globals():
                        activation_previous_object = activations_object
                        activations_object = np.concatenate((activation_previous_object,activation), axis=0)
                        activation_previous_object = activations_object

                        print(activations_object.shape)
                    else:
                        activation_previous_object = activation   

                elif count == 1:
                    activations_object = np.concatenate((activation_previous_object,activation), axis=0)
                    activations_previous_object = activations_object

                    print(activations_object.shape)                       
                        
                else:
                    activations_object = np.concatenate((activations_previous_object,activation), axis=0)
                    activations_previous_object = activations_object
                    print(activations_object.shape)
                        

            else:
                    
                if 'activations_previous_object' not in locals():
                    activations_previous_object = activations_object
                    
                activations_object = np.concatenate((activations_previous_object,activation), axis=0)
                activations_previous_object = activations_object
                print(activations_object.shape)
                    
        
            y_object.append(i)                
            print(y_object)

            count += 1               
            z += 1
            time.sleep(0.5)    

        # 사람일 경우
        else:
            playsound('object_only')
                
            print('사람이 감지가 되었습니다. {} 외의 사람들은 비켜주세요.'.format(name))
            time.sleep(.5)
    

    return count, activations_object

def check_response(response):
    while True:
        response = get_audio()
        print(response)
        if response == "" or response not in [
            "Exit",
            "exit",
            "Stop",
            "stop",
            "No",
            "no",
            "Yes",
            "yes",
            "종료",
            "종로",
            "종료 종료",
            "아니오",
            "아니요",
            "아니",
            "아니요 아니요",
            "네",
            "예",
            "내",
        ]:
            print("say again please...")
            continue
        else:
            break
    return response


def collectData(IMG_SIZE, Category):

    global y, y_object, activations, activations_object

    classes = []
    classes_object = []
    i = 0 
    i_object = 0   

    negative_responses = ["No", "no", "아니오", "아니요", "아니", "아니요 아니요"]
    esc_responses = ["종료", "종로", "종료 종료", "Stop", "stop", "Exit", "exit"]

    # 카메라 캡쳐 객체, 0=내장 카메라
    capture = cv2.VideoCapture(0)
    
    # 캡쳐 프레임 사이즈 조절
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    face_detector = cv2.CascadeClassifier('./haarcascade_xml/haarcascade_frontalface_default.xml')
    
    while True:
        name = get_name()
        

        if Category == 'Person':
            classes.append(name)
            count, _ = get_image(capture, i, IMG_SIZE, face_detector, name)
            playsound('end_person')
        else:
            classes_object.append(name)
            count, _ = get_image_object(capture, i_object, IMG_SIZE, face_detector, name)
            playsound('end_object')

        
 

        print(
            "Do you want to proceed next class(yes, no, stop, exit and etc)? (speak into the microphone)"
        )
        response = ""
        response = check_response(response)


        # 첫번째 if문은 기능 구현을 완성하지 않습니다. 방금 추가한 사람 또는 사물을 제거하는 기능입니다.
        if response in negative_responses:
        
            print("아니요")
            
            for _ in range(count):

                y.pop()

            classes.pop()
            
            continue

        elif response in esc_responses:
            print("종료")
            if Category == 'Person':
                tts = gTTS(text="안녕하세요 {} 님".format(classes[i]), lang="ko", slow=False)
                tts.save(f"saved_tts/{i}.mp3")
                break
            else:
                tts = gTTS(text="{}입니다".format(classes_object[i_object]), lang="ko", slow=False)
                tts.save(f"saved_tts/{i_object}_object.mp3")
                break

        else:
            print("네")

            if Category == 'Person':
                tts = gTTS(text="안녕하세요 {} 님".format(classes[i]), lang="ko", slow=False)
                tts.save(f"saved_tts/{i}.mp3")
                
            else:
                tts = gTTS(text="{}입니다".format(classes_object[i_object]), lang="ko", slow=False)
                tts.save(f"saved_tts/{i_object}_object.mp3")
                
            i += 1
            i_object += 1


    capture.release()
    cv2.destroyAllWindows()

    if Category == 'Person':
        y = np.asarray(y)    
    else:
        y = np.asarray(y_object) 


    print(y)    

    if Category == 'Person':
        return y, classes, activations
    else:
        return y, classes_object, activations_object

def AddData(IMG_SIZE, Category, y_load, activation_load, classes_load):

    global y, y_object, activations, activations_object

    classes = []        

    activations = activation_load
    activations_object = activation_load 

    classes = classes_load
       
    if Category == 'Person': 
        y = y_load
    else:
        y_object = y_load

    i = len(classes_load)
    negative_responses = ["No", "no", "아니오", "아니요", "아니", "아니요 아니요"]
    esc_responses = ["종료", "종로", "종료 종료", "Stop", "stop", "Exit", "exit"]

    # 카메라 캡쳐 객체, 0=내장 카메라
    capture = cv2.VideoCapture(0)
    
    # 캡쳐 프레임 사이즈 조절
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    face_detector = cv2.CascadeClassifier('./haarcascade_xml/haarcascade_frontalface_default.xml')
    
    while True:
        name = get_name()
        classes.append(name)

        if Category == 'Person':
            count = get_image(capture, i, IMG_SIZE, face_detector, name)
            playsound('end_person')
        else:
            count = get_image_object(capture, i, IMG_SIZE, face_detector, name)
            playsound('end_object')

        
        

        print(
            "Do you want to proceed next class(yes, no, stop, exit and etc)? (speak into the microphone)"
        )
        response = ""
        response = check_response(response)

        if response in negative_responses:
        
            print("아니요")
            
            # 방금 추가한 클래스를 제거하는 기능인데 , 아직 구현이 안됬습니다.
            for _ in range(count):
                # X.pop()
                
                y.pop()

            classes.pop()
            
            continue

        elif response in esc_responses:
            print("종료")
            if Category == 'Person':
                tts = gTTS(text="안녕하세요 {} 님".format(classes[i]), lang="ko", slow=False)
                tts.save(f"saved_tts/{i}.mp3")
                break
            else:
                tts = gTTS(text="{}입니다".format(classes[i]), lang="ko", slow=False)
                tts.save(f"saved_tts/{i}_object.mp3")
                break

        else:
            print("네")

            if Category == 'Person':
                tts = gTTS(text="안녕하세요 {} 님".format(classes[i]), lang="ko", slow=False)
                tts.save(f"saved_tts/{i}.mp3")
                
            else:
                tts = gTTS(text="{}입니다".format(classes[i]), lang="ko", slow=False)
                tts.save(f"saved_tts/{i}_object.mp3")
                
            i += 1


    capture.release()
    cv2.destroyAllWindows()

    if Category == 'Person':
        y = np.asarray(y)
    else:
        y = np.asarray(y_object)

    print(y)

    return y, classes, activations, activations_object

def startPredictionKNN(classifier, classes, classifier_object, classes_object):
    
    
        
    IMG_SIZE = (224, 224)
    IMG_SHAPE = IMG_SIZE + (3,)
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    base_model = getBaseModel_KNN(IMG_SHAPE)

    face_detector = cv2.CascadeClassifier('./haarcascade_xml/haarcascade_frontalface_default.xml')
    prev_class = "None"
    print("prediction starts...")

    
    
    while True:
    
        _ , frame = capture.read()
        faces = face_detector.detectMultiScale(frame, 1.03, 10, minSize=(150,150))
        arr = np.asarray(faces)
        face = False
        but = 0
        
        for (a,b,w,h) in faces:
            cv2.rectangle(frame, (a,b), (a+w,b+h), (255,0,0), 5)
        

        cv2.imshow("videFrame",frame)        
        frame_fliped = cv2.flip(frame, 1)
        
        key = cv2.waitKey(200)

        
        if (len(faces)>0):        
            frame = cv2.resize(frame[b:b+h,a:a+w], IMG_SIZE)  
            face = True   
        else:
            frame = cv2.resize(frame, IMG_SIZE)

        
        
        dataObject = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32) 
        dataObject[0] = frame        
        predict_activation = base_model(dataObject, training=False)
        predict_activation = predict_activation.numpy().reshape(1, -1)           

        #prediction 값    
        if face:                                   
            result_prob = classifier.predict_proba(predict_activation)      
        else:         
            result_prob = classifier_object.predict_proba(predict_activation)  


        print(result_prob)
                                  

        if (arr.shape == (0,) and np.max(result_prob) < 0.8 and prev_class != "background") or (arr.shape == (1, 4) and classes[np.argmax(result_prob)] == '초기화') or (arr.shape == (0,) and classes_object[np.argmax(result_prob)] == '초기화'):
               
            print(prev_class)

        # 사람이 인식이 되었을 때(face를 detect를 하고 0.9 이상 예측이 되었을 때)                   
        elif arr.shape == (1, 4) and np.max(result_prob) > 0.7 and prev_class != classes[np.argmax(result_prob)]:

            playsound(classifier.predict(predict_activation).item(0))            

            print('Face Detection Prob : {}'.format(result_prob))                        
            print(classes[np.argmax(result_prob)])
            prev_class = classes[np.argmax(result_prob)]  

        # 사물이 인식되었을 때 (face를 detect하지 못하고, 사물 0.7 이상 예측이 될 때)
        elif arr.shape == (0,) and classes_object[np.argmax(result_prob)] != '초기화' and np.max(result_prob) > 0.7 and prev_class != classes_object[np.argmax(result_prob)]:
                
            playsound(f"{classifier_object.predict(predict_activation).item(0)}_object")        
            
            print('Object Detection Prob : {}'.format(result_prob))                      
            print(classes_object[np.argmax(result_prob)])
            prev_class = classes_object[np.argmax(result_prob)]         

    return but                                            
        
if __name__ == "__main__":
    X, y, classes = collectData((224, 224))
    print(X, y, classes)

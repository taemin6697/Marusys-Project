#It's Version2 code
from playsound import playsound
import os
import cv2
import speech_recognition as sr
import time
from gtts import gTTS
import tensorflow as tf
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
import numpy as np

X=[]
y=[]

def base_model(IMG_SHAPE):#기본 모델 불러오기 
    model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224,224,3),include_top=False,weights="imagenet"
        )
    model.trainable=False
    return model

def get_prediction_layer(n_classes):
    #여긴다시봐야함
    if n_classes == 2:
        prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid")
        #_loss = tf.keras.losses.BinaryCrossentropy()
        _loss = "BinaryCrossentropy"
    else:
        prediction_layer = tf.keras.layers.Dense(n_classes, activation="softmax")
        #_loss = tf.keras.losses.sparse_categorical_crossentropy()
        _loss = "sparse_categorical_crossentropy"
    return prediction_layer,_loss

def get_model(IMG_SHAPE,prediction_layer,_loss):
    model = keras.Sequential()
    model.add(keras.Input(shape=(1, 1, 1024)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Dropout(0.5))
    model.add(prediction_layer)
    model.compile(loss=_loss,
              optimizer="rmsprop",
              metrics=["accuracy"])
    return model


def get_features_and_labels(base_model, dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = keras.applications.mobilenet_v3.preprocess_input(images)
        features = base_model.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)

def train(path,IMG_SIZE,n_classes):
    train_dataset = image_dataset_from_directory(path,image_size=IMG_SIZE,batch_size=1)
    print('1')
    conv_model = base_model(IMG_SIZE)
    print('2')
    train_feature, train_labels = get_features_and_labels(conv_model,train_dataset)
    print('3')
    pre_layer,select_loss = get_prediction_layer(n_classes)
    print('4')
    model = get_model(IMG_SIZE,pre_layer,select_loss)
    print('5')
    model.fit(train_feature, train_labels,epochs=30)
    print('6')
    model.save('saved_model.h5')
    


def get_audio():#오디오를 받습니다.
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("wait...")
        #playsound("sound_effect/wait.mp3")
        time.sleep(0.5)
        print("recording...")
        #playsound("sound_effect/recording.mp3")
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio, language="ko-KR")

        except:
 
            print("Error")
            return ""

    return said


def get_name(i): #이름을 얻어 옵니다.
    i = int(i)
    while True:
        print("What's your name? (speak into the microphone)")
        playsound("sound_effect/whatname.mp3")
        name = get_audio()
        if name == "":
            playsound("saved_tts/ask_again")
            print("say again please...")
            continue
        else:
            print(name)
            tts = gTTS(text="{} 이 이름이 맞나요".format(name), lang="ko", slow=False)
            file_path = "saved_tts/confirm{i}.mp3"
            if(os.path.exists(file_path)):
                os.remove(file_path)
            tts.save(f"saved_tts/confirm{i}.mp3")
            print('저장은 됨')
            playsound("saved_tts/confirm"+str(i)+".mp3")
            print("{} 이 이름이 맞나요?".format(name))
            #위를 대체할걸로
            #playsound("김태민 이 이름이 맞나요?로 해야함 ")
            response = get_audio()

            if response == "" or response in ["아니오", "아니요", "아니", "아리", "아니요 아니요"]:
                print("아니요")
                i+=1000
                continue

            else:
                print("네")
                break
    
    return name

def get_image(name,capture, i, IMG_SIZE):#이미지를 얻어오는 코드 입니다.
    count = 0
    while True:
        _, frame = capture.read()
        # 이미지 뒤집기
        frame_fliped = cv2.flip(frame, 1)
        # 이미지 출력
        cv2.imshow("VideoFrame", frame_fliped)

        key = cv2.waitKey(200)

        if key == 57:
            break
        if key > 0:
            playsound("sound_effect/capture.mp3")
            count += 1 
            frame = cv2.resize(frame, IMG_SIZE)
            X.append(frame)
            y.append(i)
            print('이미지 저장 실행')
            print(os.getcwd())
            if(check_set_folder(name)==True):
                saved_image(name,frame,i)
            else:
                saved_image(name,frame,random.randint(100,400))
            print('이미지 저장 실행 완료')
            i+=1
    return count

def len_of_folder():
    search_dir=search_folder()
    return len(search_dir)

def check_set_folder(name):#폴더 중복 체크
    search_dir = search_folder()
    if name in search_dir:
        return False
    else:
        return True

def search_folder():#폴더 검색
    path2 = 'data/'
    count = 0
    for(path, dir, files) in os.walk(path2):
        search_dir = dir
        if count == 0:
            break
    return search_dir

def creat_folder(name):#폴더를 생성합니다.
    search_dir = search_folder()
    if name in search_dir:
        print('파일이 이미 있습니다.')
    else:
        os.mkdir("data/"+name)

def saved_image(name,frame,i):#이미지를 생성합니다.
    i = str(i)
    #saved_tts/confirm{i}
    save_file = 'data/'+name+'/'+i+'.jpg'
    extension = os.path.splitext(save_file)[1]
    result,encoded_img = cv2.imencode(extension,frame)

    if result:
        with open(save_file,mode='w+b') as f:
            encoded_img.tofile(f)
    #print(save_file)
    #cv2.imwrite('data/'+name+'/'+i,frame)


def predict():
    model = load_model('saved_model.h5')
    model.summary()
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    IMG_SIZE = (224,224)
    conv_model = base_model(IMG_SIZE)
    classes = search_folder()
    previous_class = classes[1]
    while True:
        ret,frame = capture.read()
        frame_fliped = cv2.flip(frame, 1)
        cv2.imshow("VideoFrame",frame_fliped)
        key = cv2.waitKey(50)
        if key > 4:
            break
        frame = cv2.resize(frame,IMG_SIZE)
        frame = np.array(frame)
        frame = frame.reshape(1, 224, 224, 3)
        #frame = keras.applications.mobilenet_v3.preprocess_input(frame)
        frame = conv_model.predict(frame)         
        pre = model.predict(frame)
        print(np.max(pre))


        if(np.max(pre)<0.7 and previous_class != 'background'):
            previous_class='background'
            print("background")
        elif(np.max(pre)>0.8 and previous_class != classes[np.argmax(pre)]):
            if('0' in classes[np.argmax(pre)]): 
                previous_class = classes[np.argmax(pre)]
                print(classes[np.argmax(pre)]+'입니다.')
            else:
                previous_class = classes[np.argmax(pre)]
                print(classes[np.argmax(pre)]+'님 안녕하세요')


def Add_object(IMG_SIZE): #데이터를 추가하는 코드 입니다.
    global X,y
    classes = []
    i = 0
    negative_responses = ["아니오", "아니요", "아니", "아니요 아니요"]
    esc_responses = ["종료", "종로", "종료 종료"]

    #카메라 객체 생성
    capture = cv2.VideoCapture(0)

    #카메라 사이즈 조절
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        name = get_name(i)
        name = name+'0'
        classes.append(name)
        creat_folder(name)
        count = get_image(name,capture,i,IMG_SIZE)     
        i+=1

def Add_data(IMG_SIZE): #데이터를 추가하는 코드 입니다.
    global X,y
    classes = []
    i = 0
    negative_responses = ["아니오", "아니요", "아니", "아니요 아니요"]
    esc_responses = ["종료", "종로", "종료 종료"]

    #카메라 객체 생성
    capture = cv2.VideoCapture(0)

    #카메라 사이즈 조절
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        name = get_name(i)
        classes.append(name)
        creat_folder(name)
        count = get_image(name,capture,i,IMG_SIZE)     
        i+=1

IMG_SIZE = (224,224)
print(len_of_folder())
su = len_of_folder()
name_list = []
#Add_object(IMG_SIZE)
#Add_data(IMG_SIZE)
train('data/',IMG_SIZE,su)
#print("완료!")
#path = 'data/retest/'
#model = load_model('saved_model.h5')
#model.summary()
#train_dataset = image_dataset_from_directory(path,image_size=IMG_SIZE,batch_size=1)
#conv_model = base_model(IMG_SIZE)
#test_feature, test_labels = get_features_and_labels(conv_model,train_dataset)
#score = model.evaluate(test_feature,test_labels,verbose=0)
#print(score)
 
predict()

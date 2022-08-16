import numpy as np
from camera import collectData, AddData, startPredictionKNN
from model import train_KNN, saveClassandLabel, saveClassandLabel_object
import pygame
import os

IMG_SIZE = (224, 224)

def printManual():

    playsound('instruction')

    if 'classifier' or 'classifier_object' in globals():
        print("Append Data " + "=" * 22)
    else:
        print("Choose a mode " + "=" * 22)

    print("%30s%5s" % ("[Person init or add]:", "1"))    
    print("%30s%5s" % ("[Object init or add]:", "2"))     
    print("%30s%5s" % ("[Prediction]:", "3"))
    print("%30s%5s" % ("[Exit]:", "9"))
    print("=" * 36)

    

def playsound(file):

    pygame.mixer.init()
    pygame.mixer.music.load("saved_tts/{}.mp3".format(file))
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy() == True:
        continue

def get_labels_from_file(label_path):
    classes = []

    f = open(label_path, "r", encoding="cp949")
    while True:
        line = f.readline()
        if not line:
            break
        classes.append(line.rstrip("\n"))
    f.close()

    return classes


def get_activations_y_classes_from_file(activations_path,y_path, classes_path):
    y_load = []
    classes_load = []

    f = open(classes_path, "r", encoding="cp949")
    while True:
        line = f.readline()
        if not line:
            break
        classes_load.append(line.rstrip("\n"))
    f.close()

    f = open(y_path, "r", encoding="cp949")
    while True:
        line = f.readline()
        if not line:
            break
        y_load.append(line.rstrip("\n"))
    f.close()

    activation_load = np.load(activations_path)
    

    return y_load, activation_load, classes_load

def initialization_person():
    global classifier

    y, classes, activations = collectData(IMG_SIZE, 'Person')
    np.save('./saved_activations/activations_saved', activations)

    saveClassandLabel(classes, y)
    classifier = train_KNN(activations, y)

    return classifier

def initialization_object():
    global classifier_object

    y, classes, activations_object = collectData(IMG_SIZE, 'Object')
    np.save('./saved_activations/activations_saved_object', activations_object)                
    saveClassandLabel_object(classes, y)
    classifier_object = train_KNN(activations_object, y) 

    return classifier_object


def append_data_person():

    y_path = "./saved_model/ylist.txt"
    activations_path = "./saved_activations/activations_saved.npy"
    classes_path = "./saved_model/labels.txt"

    y_load, activation_load, classes_load  = get_activations_y_classes_from_file(activations_path,y_path, classes_path)            
    y, classes, activations, _ = AddData(IMG_SIZE,'Person', y_load, activation_load, classes_load)

    np.save('./saved_activations/activations_saved', activations)
    saveClassandLabel(classes, y)
    classifier = train_KNN(activations, y) 

    return classifier

def append_data_object():

    y_path = "./saved_model/ylist_object.txt"
    activations_path = "./saved_activations/activations_saved_object.npy"
    classes_path = "./saved_model/labels_object.txt"

    y_load, activation_load, classes_load  = get_activations_y_classes_from_file(activations_path,y_path, classes_path)            
    y, classes, _ , activations_object = AddData(IMG_SIZE,'Object', y_load, activation_load, classes_load)

    np.save('./saved_activations/activations_saved_object', activations_object)
    saveClassandLabel_object(classes, y)
    classifier_object = train_KNN(activations_object, y)  

    return classifier_object

def load_prediction():
    Load = False
    Load_object = False

    print('loading from local files ..')        

    if os.path.isfile('./saved_activations/activations_saved.npy') == True: 
        print('Face model exists!')
        Load = True
        y_path = "./saved_model/ylist.txt"     
        label_path = "./saved_model/labels.txt"
        classes = get_labels_from_file(label_path)
        activations_path = "./saved_activations/activations_saved.npy"
        y_load, activation_load, _ = get_activations_y_classes_from_file(activations_path,y_path, label_path)
    else:
        print('Face model is empty. From the very beginning')       
        Load = False
        y_path = "./saved_model/init_ylist.txt"     
        label_path = "./saved_model/init_labels.txt"
        classes = get_labels_from_file(label_path)
        activations_path = "./saved_activations/init_activations.npy"
        y_load, activation_load, _ = get_activations_y_classes_from_file(activations_path,y_path, label_path)   

    if os.path.isfile('./saved_activations/activations_saved_object.npy') == True: 
        print('Object model exists!')
        Load_object = True
        y_path = "./saved_model/ylist_object.txt"     
        label_path = "./saved_model/labels_object.txt"
        classes_object = get_labels_from_file(label_path)
        activations_path = "./saved_activations/activations_saved_object.npy"
        y_load_object, activation_load_object, _ = get_activations_y_classes_from_file(activations_path,y_path, label_path)
    else:    
        print('Object model is empty. From the very beginning')
        Load_object = False      
        y_path_object = "./saved_model/init_ylist_object.txt"     
        label_path_object = "./saved_model/init_labels_object.txt"
        classes_object = get_labels_from_file(label_path_object)
        activations_path_object = "./saved_activations/init_activations_saved_object.npy"        

        y_load_object, activation_load_object, _ = get_activations_y_classes_from_file(activations_path_object,y_path_object, label_path_object)
    

    classifier = train_KNN(activation_load, y_load)
    classifier_object = train_KNN(activation_load_object, y_load_object)

    button = startPredictionKNN(classifier,classes, classifier_object, classes_object)

    return Load, Load_object, button





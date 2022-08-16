from camera import startPredictionKNN
from setting import printManual, playsound, initialization_person, initialization_object, append_data_person, append_data_object, load_prediction
import taemin
#from Button import PrepareButton, ButtonPushed
from voice import bot_audio, excute_voice_command, delete_voice_command, add_voice_command
#from dashpy import dashReset, dashmacSet, dashConnect
from voicecmd import startVoiceCommand
#from dashlib import dashlib
import time

IMG_SIZE = (224, 224)

button = [13, 18, 31, 29, 36]
LED = [11, 15, 3]

def main():
    
    global classifier
    global classifier_object
    #printManula()은 현재 인터럽트가 불가 예시 코드임
    printManual()
    mode = int(input("추가하실거면 1번을 눌러주세요"))
    if mode == 1:
        print("사람이나 물건을 추가합니다.")
        #사람 or 물건 추가 코드

    while(True):
        playsound('start')
        Load, Load_object, but = load_prediction()                     
            #여기엔 인터럽트를 추가하면됨           

if __name__ == "__main__":
    main()

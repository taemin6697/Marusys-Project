#from dashpy import dashmacSet, Dash_Voice_Command, dashScanning, dashFaceCenter, dashReset, dashConnect, ComeOn
from voice import excute_voice_command, bot_audio
#import cv2
#import numpy as np
#from Button import PrepareButton, ButtonPushed
# face_detector = cv2.CascadeClassifier('./haarcascade_xml/haarcascade_frontalface_default.xml')
# button = [13, 18, 31, 29, 36] # buttons 1, 2, 3, 4
voice_command = ""


def startVoiceCommand(dashmac,bot):
    global voice_command
    #dashmac = dashmacSet()
    #bot = dashConnect(dashmac)
#     global face_detector, voice_command
#     IMG_SIZE = (224, 224)
#     IMG_SHAPE = IMG_SIZE + (3,)
#     capture = cv2.VideoCapture(0)
#     capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    
    
    
    
    while True:
        if dashmac is None:
            bot = dashReset()
        said = bot_audio()
        if said != '':
            voice_command = excute_voice_command(said, bot)
            


#         _ , frame = capture.read()
#         faces = face_detector.detectMultiScale(frame, 1.03, 10, minSize=(150,150))
#         arr = np.asarray(faces)
#         face = False
#         but = 0
# 
#         for (a,b,w,h) in faces:
#             cv2.rectangle(frame, (a,b), (a+w,b+h), (255,0,0), 5)
#         
# 
#         PrepareButton(button)
# 
#         # 얼굴을 자를지 안자를지 / 1명 여러명, none by 길이를 재서
#         
#         frame_fliped = cv2.flip(frame, 1)
# 
#         cv2.imshow("VideoFrame", frame_fliped)
# 
#         key = cv2.waitKey(200)

        # if ButtonPushed(button[4]):
        #     break

#         if ButtonPushed(button[0]):
#             but = 0            
#             break
# 
#         if ButtonPushed(button[1]):
#             but = 1            
#             break
# 
#         if ButtonPushed(button[2]):
#             but = 2           
#             break

#         said = bot_audio()
#         if said != '':
#             voice_command = excute_voice_command(said, bot)
#             
        
#         if len(faces) > 0 :
#             if dashScan:
#                 stop(bot)
#             if voice_command == 'ComeOn':                
#                 dashCenter = True
#                 if dashCenter:
#                     dashCenter = dashFaceCenter(bot, 10)
#                 else :
#                     ComeOn(bot,faces)
#             
#         else :
#             if voice_command == 'ComeOn':
#                 dashScan = True
#                 if dashScan:
#                     dashScan = dashScanning(bot)

# 
# if __name__ == "__main__":
#     main()
# 
#     

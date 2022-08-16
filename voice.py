import speech_recognition as sr
import time
#from dashpy import Dash_Voice_Command

classes = []
command_sound = {"ComeOn" : ['이리 와', '이리로 와', '일로 와', '일루 와', '이리와', '이리로와', '일로와', '일루와'],
                "GoAway" : ['저리 가', '저리로 가', '절로 가', '절루 가', '저리가', '저리로가', '절로가', '절루가'], 
                "Spin" : ['도라', '돌아'],
                "Left" : ['왼쪽', '좌측'],
                "Right" : ['오른쪽', '우측'],
                "Stop" : ['그만', '멈춰']}

def bot_audio():
    timeout = 2
    r = sr.Recognizer()
    r.energy_threshold = 1000       #라즈베리파이에서 필요함
    print("Get Audio")
    with sr.Microphone() as source:
        audio = r.adjust_for_ambient_noise(source)      #라즈베리파이에서 노이즈 제거
        #audio = r.listen(source)            #phrase_time_limit = 2 test
        st = time.time()
        try :
            audio = r.listen(source, timeout = timeout, phrase_time_limit= 2)
        except:
            pass
        ed = time.time()
        print("listen : ",ed-st)
        #listen(self, source, timeout=None, phrase_time_limit=None, snowboy_configuration=None)
        said = ""

        try:
            said = r.recognize_google(audio, language="ko-KR")
            #1st test
            #recognize_bing(): Microsoft Bing Speech
            #recognize_google(): Google Web Speech API
            #recognize_google_cloud(): Google Cloud Speech - requires installation of the google-cloud-speech package
            #recognize_houndify(): Houndify by SoundHound
            #recognize_ibm(): IBM Speech to Text
            #recognize_sphinx(): CMU Sphinx - requires installing PocketSphinx
            #recognize_wit(): Wit.ai

        except:
            print("Error")
            return ""
        
        #if said != "":
            #excute_voice_command(said, bot)
    #print(said)
    return said

def get_class():
    global classes
    for i in range(0,5):

        print("Speak into the microphone")
        
        name = bot_audio()
        if name == "":
            print("say again please...")
            #continue
        else:
            print(name)
            if name not in classes:
                classes.append(name)
            print(classes)
    return classes


def delete_voice_command(audio,key):
    print("Delete : ", command_sound[key])
    #name = get_audio()
    for i in range(len(command_sound[key])):
        if audio == command_sound[key][i]:
            del command_sound[key][i]

def add_voice_command(audio, key):
    print("Add : ", command_sound[key])
    #name = get_audio()
    if audio not in command_sound[key]:
        command_sound[key].append(audio)


def excute_voice_command(audio, bot):
    print("Excute Voice Command")
    #name = get_audio()

    for i in command_sound.keys():
        if audio in command_sound[i]:
            voice_command = i
            Dash_Voice_Command(bot, i)    
            print("Command : ", i)
            return voice_command


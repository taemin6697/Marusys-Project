#import RPi.GPIO as GPIO
#import time


#def PrepareButton(button):

#    try:
#        time.sleep(0.2)
                
#    finally:
#        GPIO.cleanup(button)

#    # Prepare GPIO Button
#    GPIO.setwarnings(False)
#    GPIO.setmode(GPIO.BOARD)
#    GPIO.setup(button, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) 

#def ButtonPushed(button):

#    if GPIO.input(button) == GPIO.HIGH:
#       return True
    
# The eye aspect ratio (EAR) threshold that the EAR needs
# to be below to be considered closed.
EYE_AR_THRESH = 0.26
# The consecutive frames that the eyes need to be closed to 
# indicate a blink: dot
EYE_AR_CONSEC_FRAMES = 4
# Consec frames the EAR must be below the threshold: dash
EYE_AR_CONSEC_FRAMES_CLOSED = 12
# Consec frames the eye must be open the threshold: /
PAUSE_CONSEC_FRAMES = 25
# Consec frames the eye must be open the threshold to indicated a pause
# between words. This is added with PAUSE_CONSEC_FRAMES to detect pause
WORD_PAUSE_CONSEC_FRAMES = 35
# Consec frames eyes must be closed to exit the program
BREAK_LOOP_FRAMES = 60
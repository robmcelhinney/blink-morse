from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import morse_code


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	eye_ar = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return eye_ar

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
help="path to input video file")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.26
EYE_AR_CONSEC_FRAMES = 4
EYE_AR_CONSEC_FRAMES_CLOSED = 20
PAUSE_CONSEC_FRAMES = 40
WORD_PAUSE_CONSEC_FRAMES = 90
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
EYES_OPEN_COUNTER = 0
CLOSED_EYES = False
PAUSED = False
WORD_PAUSE = False
morse_str = ""


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)


# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		left_eye_ar = eye_aspect_ratio(leftEye)
		right_eye_ar = eye_aspect_ratio(rightEye)
		# average the eye aspect ratio together for both eyes
		eye_ar = (left_eye_ar + right_eye_ar) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if eye_ar < EYE_AR_THRESH:
			COUNTER += 1
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				CLOSED_EYES = True
			# if PAUSED and EYES_OPEN_COUNTER >= PAUSE_CONSEC_FRAMES:
			# 	morse_str += '/'
			# 	PAUSED = False
			# 	CLOSED_EYES = False
			# 	EYES_OPEN_COUNTER = 0
		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			EYES_OPEN_COUNTER += 1
			if COUNTER >= EYE_AR_CONSEC_FRAMES_CLOSED:
				morse_str += "-"
				# reset the eye frame counter
				COUNTER = 0
				CLOSED_EYES = False
				PAUSED = True
				EYES_OPEN_COUNTER = 0
			elif CLOSED_EYES:
				morse_str += "."
				TOTAL += 1
				COUNTER = 1
				CLOSED_EYES = False
				PAUSED = True
				EYES_OPEN_COUNTER = 0
			elif PAUSED and EYES_OPEN_COUNTER >= PAUSE_CONSEC_FRAMES:
				morse_str += '/'
				PAUSED = False
				WORD_PAUSE = True
				CLOSED_EYES = False
				EYES_OPEN_COUNTER = 0
			elif WORD_PAUSE and EYES_OPEN_COUNTER >= WORD_PAUSE_CONSEC_FRAMES:
				# Remove '/' as it was a word pause not a char pause
				morse_str = morse_str[:-1]
				morse_str += '/-..-./'
				WORD_PAUSE = False
				CLOSED_EYES = False
				EYES_OPEN_COUNTER = 0

		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(eye_ar), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "morse_str: {}".format(morse_str), (10, 300),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		print('morse_str: {}\r'.format(morse_str), end="")

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


print("morse_str: ", morse_str)
print("from_morse(morse_str): ", morse_code.from_morse(morse_str))

# HELLO WORLD
# .... . .-.. .-.. --- / .-- --- .-. .-.. -..


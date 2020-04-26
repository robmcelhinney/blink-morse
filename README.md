# Blink Morse
Eye blink detection of morse code that types.

## Demo
![Blinking Hello World](demo/sample.gif)

## Requirements
Download dlib pre-trained facial landmark predictor available at [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

Uncompress the file and store the .dat file in the same dir as detect_blinks.py


Install requirements:  
  ```
  pip install -r requirements.txt
  ```
  
## Using from command line
    python detect_blinks.py -p shape_predictor_68_face_landmarks.dat

To exit the program:
    Type ']' when webcam in focus or close eyes for constants.BREAK_LOOP_FRAMES frames.

### Change variables
I'd suggest you change the variables in constants.py so your blinks are better recognised.

By editing the variables you can give yourself more time before the next blink/pause is detected making morse code easier to input. 

## Built With

* dlib - C++ toolkit containing machine learning algorithms.
* OpenCV - Library mainly aimed at real-time computer vision.
* imutils - Series of convenience functions to make basic image processing functions.


## Acknowledgments
Blink detection based off of tutorial from [pyimagesearch](https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib).

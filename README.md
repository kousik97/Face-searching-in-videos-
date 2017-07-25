# Face-searching-in-videos-
Python script to search people in videos.
Takes a video and facial image as input and outputs whether the face was found in the video along with more information like time, duration, et al.


## Requirements 
1. python2
2. dlib
3. opencv
4. avconv

All codes are tested in a container built from Ubuntu 14.04 CPU image downloaded from floydhub(see link below)

## Demo
1. Download all the required weight files from [drive]() and copy them in the directory containing demo.py
2. Run python demo.py -v full/path/to/video -i full/path/to/image

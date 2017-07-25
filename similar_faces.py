
from __future__ import division
import os
import shutil
import sys
import dlib
from skimage import io
import argparse
import cv2
import numpy as np
#import face_recognition
result_dir="./Image_Search"
predictor_path = "./shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat"
#result_file="/home2/rajib/face/output.pkl"

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)



def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    import numpy as np
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare)



def face_feature(frame):
    #print("Processing file: {}".format(frame))
    #img = io.imread(frame)
    image1 = cv2.imread(frame)
    height, width = image1.shape[:2]
    #print((height,width))
    if width > 500:
        r = 500.0 / image1.shape[1]
        dim = (500, int(image1.shape[0] * r))
        img2 = cv2.resize(image1, dim, interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        #print(image.shape)
    else:
        img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)


    #win.clear_overlay()
    #win.set_image(img)
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    # Now process each face we found.
    if len(dets) != 0: 
        distance = []
        for k, d in enumerate(dets):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                #k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = sp(img, d)
            # Draw the face landmarks on the screen so we can see what face is currently being processed.
            #win.clear_overlay()
            #win.add_overlay(d)
            #win.add_overlay(shape)

            face_descriptor = facerec.compute_face_descriptor(img, shape)
            a=np.array(face_descriptor)
            distance.append(face_distance(query,a))
        minimum_dist = min(distance)
        if minimum_dist < threshold:
            img1=cv2.imread(frame)
            save_path=result_dir+"/"+frame.split("/")[-1]
            cv2.imwrite(save_path,img1)
        else:
            pass
    else:
        pass    

def query_feature(frame):
    image1 = cv2.imread(frame)
    height, width = image1.shape[:2]
    if width > 500:
        r = 500.0 / image1.shape[1]
        dim = (500, int(image1.shape[0] * r))
        img2 = cv2.resize(image1, dim, interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        #print(image.shape)
    else:
        img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    
    dets = detector(img, 1) 
    if len(dets) != 0:
        for k, d in enumerate(dets):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                #k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = sp(img, d)
            # Draw the face landmarks on the screen so we can see what face is currently being processed.
            #win.clear_overlay()
            #win.add_overlay(d)
            #win.add_overlay(shape)

            face_descriptor = facerec.compute_face_descriptor(img, shape)
            a=np.array(face_descriptor)
            return(a)
    else:
        print("No face detected in Query image")
        
       
def parse_args():
    """Parse input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Get a shape mesh (t-pose)')
    parser.add_argument('--model_dir', type=str, help='model dir', required=True)
    parser.add_argument('--input', type=str, help='Input dir of images', required=True)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    """ Entry point """
    args = parse_args()
    threshold = 0.55
    try:
        shutil.rmtree(result_dir, ignore_errors=True)
        os.mkdir(result_dir)
    except:
        os.mkdir(result_dir)
    image_path = args.input
    query = query_feature(image_path)
    file_path = args.model_dir
    for root, subdirs, files in os.walk(file_path):
        for f in files:
            img_path = os.path.join(root, f)
            face_feature(img_path)



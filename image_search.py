import cv2 
import os 
from skimage import io
import argparse


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Video to find the image")
    ap.add_argument("-i","--image",help="Image to search")
    args=vars(ap.parse_args())
    if args["video"] and os.path.isfile(args["video"]):
     print("Video Found")
     if args["image"] and os.path.isfile(args["image"]):
       print("Image Found")
       if(os.path.isdir("./Temp_Dir")):
           os.system("rm -rf Temp_Dir")
       os.system("mkdir Temp_Dir")
       os.chdir("Temp_Dir")
       video_name=((args["video"].split('/')[-1]).split('.')[0])
       path= video_name+"-" +"%4d.jpg"
       command="avconv -i "+str(args["video"])+" -r 2 "+path
       os.system(command)
       os.chdir("..")
       command="python similar_faces.py --model_dir ./Temp_Dir --input "+args["image"]
       print(command)
       os.system(command)
       timelines=[] 
       for x in os.listdir("Image_Search"):
           timelines.append(int((x.split('-')[1]).split('.')[0])*0.5)
       if(len(timelines)==0):
          print("The image was not found")
       else:
          print("The image was found in the video "+str(len(timelines))+"times")
          print("The first time the images was found in the video was at "+str(min(timelines))+"s")
          print("The last time the image was found in the video was at "+str(max(timelines))+"s")
     else:
       print("Image Not found")
    else:
       print("Video Not found")
  
       

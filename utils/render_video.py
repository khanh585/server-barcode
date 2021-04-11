from moviepy.editor import *
import time

def set_fps(pathVideo, fps, tail):

    t0 = time.time()

    clip = VideoFileClip(pathVideo)
   
    if tail != '':
        pathVideo = pathVideo.replace('.', tail+'.')
    
    clip.write_videofile(pathVideo, fps = fps)

    print(time.time() - t0)
    return pathVideo

def resize_video(pathVideo, size, tail):

    t0 = time.time()

    clip = VideoFileClip(pathVideo)
    clip.resize(size) 
    if tail != '':
        pathVideo = pathVideo.replace('.', tail+'.')
    clip.write_videofile(pathVideo, bitrate='3700k')

    print(time.time() - t0)
    return pathVideo
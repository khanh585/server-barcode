import cv2
import numpy as np 
import glob
from pyzbar.pyzbar import decode
from matplotlib import pyplot as plt



COLOR = (250,0,10)
COLORT = (25,0,10)


def detect_barcode(img):
    codes = decode(img)
    text = ''
    for barCode in codes:
        # get message
        text = barCode.data.decode('utf-8')

    return text

def get_line(img):
    y = len(img)
    x = len(img[0])
    y1 = int(y / 6 * 1) 
    y2 = int(y / 6 * 3) 
    y3 = int(y / 6 * 5)

    cv2.line(img, (0, y1), (x,y1), COLOR, 1)
    cv2.line(img, (0, y2), (x,y2), COLOR, 1)
    cv2.line(img, (0, y3), (x,y3), COLOR, 1)
    
    return [list(img[y1+3]), list(img[y2+3]), list(img[y3-3])], img

def line_to_barcode(line):
    code = []
    for i in range(40):
        code.append(line)
    return np.array(code)

def crop_black(line):
    line_f = np.array(line)

    line_f[line_f < 120] = 0
    line_f[line_f != 0] = 255

    line_f = list(line_f)

    head = 0
    while True:
        if line_f[0] == 255:
            break
        line_f.pop(0)
        line.pop(0)
        head += 1
    tail = 0
    while True:
        if line_f[-1] == 255:
            break
        line_f.pop()
        line.pop()
        tail += 1
    return line, head, tail

def crop_white(line):
    line_f = list(line)
    line = list(line)

    head = 0
    while True:
        if line_f[0] != 255:
            break
        line_f.pop(0)
        line.pop(0)
        head += 1
    tail = 0
    while True:
        if line_f[-1] != 255:
            break
        line_f.pop()
        line.pop()
        tail += 1

    return line, head, tail



    return thresh2

def rotate_img(img):
    height = img.shape[0]
    width = img.shape[1]
    images = []
    if height > width:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        images.append(img)
        img = cv2.rotate(img, cv2.ROTATE_180)
        images.append(img)
    return images


def process_error(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgs = rotate_img(img)
    text = ''
    out = False
    
    for image in imgs:
        lines, _ = get_line(image)
        for line in lines:
            line, _, _ = crop_black(line)
            line = np.array(line)
            line[line > 120] = 255
            line[line != 255] = 0
            barcode = line_to_barcode(line)
            text = detect_barcode(barcode)
            if text != '':
                break
        if out:
            break

    return text


def highlingh_barcode(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_blur = cv2.GaussianBlur(im_gray,(3,3),1)
    ave = int(np.average(im_blur)) + 41
    _ ,im_thres = cv2.threshold(im_blur, ave, 255, cv2.THRESH_BINARY)

    # clean noise
    for i in range(5):
        im_blur = cv2.GaussianBlur(im_thres,(7,7),1)
        ave = int(np.average(im_blur)) + 41
        _ ,im_thres = cv2.threshold(im_blur, ave, 255, cv2.THRESH_BINARY)

    return im_thres, ave

def read_barcode(img):
    text = detect_barcode(img)
    if text == '':
        text = process_error(img)
    return text

        
            

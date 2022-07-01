import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
from pdf2image import convert_from_path
from io import BytesIO
from joblib import load
import imutils
from imutils.contours import sort_contours
import argparse
from pathlib import Path

def segment(page):
    img = np.array(page)

    plotting = plt.imshow(img,cmap='gray')
    #plt.show()
    
    #thresholding the image to a binary image
    thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)
    #inverting the image 
    img_bin = 255-img_bin
    cv2.imwrite('/Users/YOURPATH/cv_inverted.png',img_bin)
    #Plotting the image to see the output

    # Length(width) of kernel as 100th of total width
    kernel_len = np.array(img).shape[1]//100
    # Defining a vertical kernel to detect all vertical lines of image 
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    #Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    cv2.imwrite("vertical.jpg",vertical_lines)
    #Plot the generated image
    plotting = plt.imshow(image_1,cmap='gray')
    #plt.show()

    #Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    cv2.imwrite("horizontal.jpg",horizontal_lines)
    #Plot the generated image
    plotting = plt.imshow(image_2,cmap='gray')
    #plt.show()

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 1, horizontal_lines, 1, 0.0)
    #Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("img_vh.jpg", img_vh)
    bitxor = cv2.bitwise_xor(img,img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    #Plotting the generated image
    plotting = plt.imshow(img_vh,cmap='gray')


    # Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def sort_contours(cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    #Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    #Get mean of heights
    mean = np.mean(heights)

    #Create list box to store all boxes in  
    box = []
    # Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w<1000 and h<500):
            image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            box.append([x,y,w,h])

    plotting = plt.imshow(image,cmap="gray")
    plt.show()

    #Creating two lists to define row and column in which cell is located
    row=[]
    column=[]
    j=0
    #Sorting the boxes to their respective row and column
    for i in range(len(box)):
        if(i==0):
            column.append(box[i])
            previous=box[i]
        else:
            if(box[i][1]<=previous[1]+mean/2):
                column.append(box[i])
                previous=box[i]
                if(i==len(box)-1):
                    row.append(column)
            else:
                row.append(column)
                column=[]
                previous = box[i]
                column.append(box[i])

    #calculating maximum number of cells
    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol

    #Retrieving the center of each column
    center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]
    center=np.array(center)
    center.sort()

    #Regarding the distance to the columns center, the boxes are arranged in respective order
    finalboxes = []
    for i in range(len(row)):
        lis=[]
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)
    #from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
    outer=[]
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner=""
            if(len(finalboxes[i][j])==0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y1,x1,w1,h1 = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
                    finalimg = bitnot[x1:x1+h1, y1:y1+w1]
                    thresh = cv2.threshold(finalimg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

                    # Remove horizontal
                    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
                    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
                    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                    for c in cnts:
                        cv2.drawContours(finalimg, [c], -1, (255,255,255), 2)


                    # Remove vertical
                    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,30))
                    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
                    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                    for c in cnts:
                        cv2.drawContours(finalimg, [c], -1, (255,255,255), 2)
                    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
                    finalimg = 255 - cv2.morphologyEx(255 - finalimg, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

                    Path(f"boxes/{j}").mkdir(exist_ok=True)
                    cv2.imwrite(f"boxes/{j}/{i}.jpg", finalimg)
                    inner = ocr(finalimg)
                    
                outer.append(inner)


    #Creating a dataframe of the generated OCR list
    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(row),countcol))
    data = dataframe.style.set_properties(align="left")
    #Converting it in a excel-file
    data.to_excel("output.xlsx")

def ocr(img):
    model = load("mnist.sav")
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # perform edge detection, find contours in the edge map, and sort the
    # resulting contours from left-to-right
    edged = cv2.Canny(blurred, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return ""
    cnts = sort_contours(cnts, method="left-to-right")[0]
    # initialize the list of contour bounding boxes and associated
    # characters that we'll be OCR'ing
    chars = []

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # filter out bounding boxes, ensuring they are neither too small
        # nor too large
        if (w >= 5 and w <= img.shape[0]-1) and (h >= 15 and h <= img.shape[1]-1):
            # extract the character and threshold it to make the character
            # appear as *white* (foreground) on a *black* background, then
            # grab the width and height of the thresholded image
            roi = img[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape
            # if the width is greater than the height, resize along the
            # width dimension
            if tW > tH:
                thresh = imutils.resize(thresh, width=28)
            # otherwise, resize along the height
            else:
                thresh = imutils.resize(thresh, height=28)

            # re-grab the image dimensions (now that its been resized)
            # and then determine how much we need to pad the width and
            # height such that our image will 28x28
            (tH, tW) = thresh.shape
            dX = int(max(0, 28 - tW) / 2.0)
            dY = int(max(0, 28 - tH) / 2.0)
            # pad the image and force 28x28 dimensions
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0))
            
            padded = cv2.resize(padded, (28, 28))
            # prepare the padded image for classification via our
            # handwriting OCR model
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)
            # update our list of characters that will be OCR'd
            chars.append((padded, (x, y, w, h)))
    # extract the bounding box locations and padded characters
    boxes = [b[1] for b in chars]
    chars = np.array([np.multiply(c[0], 255) for c in chars], dtype="float32")
    # for char in chars:
    #     plotting = plt.imshow(char.reshape((28,28)),cmap='gray')
    #     plt.show()
    chars = chars.reshape((-1, 784))
    
    # OCR the characters using our handwriting recognition model
    if len(chars) == 0:
        return ""
    else:
        preds = model.predict(chars)
        #for (label, (x, y, w, h)) in zip(preds, boxes):
        #    # draw the prediction on the image
        #    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #    cv2.putText(image, str(label), (x - 10, y - 10),
        #        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        #    # show the image
        #    cv2.imshow("Image", img)
        return "".join([str(i) for i in preds])

def convert(path):
    pages = convert_from_path(path, grayscale = True)

    for page in pages:
        segment(page)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("path", metavar="P", type=str, help="Path to pdf")
    args = parser.parse_args()
    convert(args.path)

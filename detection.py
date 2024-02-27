import cv2
import matplotlib.pyplot as plt
import numpy as np

def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(data_path, clf):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (ML_Models_pred.txt), the format is the same as Sample.txt.
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)

    f = open(data_path, 'r')
    text = []
    for line in f:
        position = line.split()
        text.append(position)
    f.close()


    cap = cv2.VideoCapture("./data/detect/video001.gif")
    fo = open("ML_Models_pred.txt", "w")
    n = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break
        print('第'+str(n)+'張frame')
        if n != 1:
          fo.writelines('\n')
        n +=1
        result=[]
        for i in range(1,77):
            x1 = text[i][0]
            y1 = text[i][1]
            x2 = text[i][2]
            y2 = text[i][3]
            x3 = text[i][4]
            y3 = text[i][5]
            x4 = text[i][6]
            y4 = text[i][7]
            cropped = crop(x1,y1,x2,y2,x3,y3,x4,y4,frame)
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            cropped = cv2.resize(cropped, (16, 36))
            flat = cropped.flatten()
            input = np.array(flat).reshape(1,-1)
            result.append(str(clf.classify(input))+' ')
        fo.writelines(result)
        # if frame is read correctly ret is True
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    fo.close()
  
    # End your code (Part 4)

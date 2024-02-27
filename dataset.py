import os
import cv2

def load_images(data_path):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    car_file_name = os.path.join(dir_path,data_path,'car')
    non_car_file_name = os.path.join(dir_path,data_path,'non-car')
    store_img=[]
    for item in os.listdir(car_file_name):
        img = cv2.imread(car_file_name+'/'+item)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (36, 16))
        store_img.append([img,1])

    for item in os.listdir(non_car_file_name):
      img = cv2.imread(non_car_file_name+'/'+item)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      img = cv2.resize(img, (36, 16))
      store_img.append([img,0])

    dataset = store_img
    # End your code (Part 1)
    return dataset

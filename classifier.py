import numpy as np, cv2, os
from tqdm import tqdm
import keras
from keras.models import load_model
import datetime
#the images are 200x200 pixles, too big for Keras
#resize them to this size
img_size = 60

#-------------get train/test data-----------------
#get data
classes = ['DR','FT','LP']
classifier = load_model('./Image matching/Balnc/myCode/models/rec1/rec1_model.h5')
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.preprocessing import image
IMAGE_FOLDER = './train/basketball/'
images = os.listdir(IMAGE_FOLDER)
count=0
a = datetime.datetime.now().replace(microsecond=0)
for filename in images:
    img = cv2.imread(IMAGE_FOLDER+ filename)
    test_image = image.load_img(IMAGE_FOLDER+ filename, target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    print(filename + " -> " + classes[np.argmax(result)])
    if classes[np.argmax(result)] == 'DR':
        cv2.imwrite('./dribble/'+filename,img)
        count+=1
    elif classes[np.argmax(result)] == 'LP':
        cv2.imwrite('./layup/'+filename,img)
    else:
        cv2.imwrite('./freethrow/'+filename,img)
print("\n correct count = "+ str(count) +"\n total count = "+str(len(images))+"\n")
print("Percentage:" + str((count)/len(images)*100)+"%")
print("\n")
b = datetime.datetime.now().replace(microsecond=0)
print("time for testing: " + str(b-a) )

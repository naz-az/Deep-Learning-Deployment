import io
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("Best_pretrained.h5")

class_names = ['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon', 'hamburger', 'ice_cream', 'pizza', 'ramen', 'steak', 'sushi']


with open("Test/fried-rice-download.jpg", "rb") as file:
        image_bytes = file.read()
        pillow_img = Image.open(io.BytesIO(image_bytes)).convert('L')


data = np.asarray(pillow_img)
# data = data / 255.0
data = data[np.newaxis, ..., np.newaxis]
# --> [1, x, y, 1]
data = tf.image.resize(data, [224, 224])


predictions = model(data)
# predictions = tf.nn.softmax(predictions)
pred0 = predictions[0]
label0 = np.argmax(pred0)
class_name = class_names[label0]
print(class_name)

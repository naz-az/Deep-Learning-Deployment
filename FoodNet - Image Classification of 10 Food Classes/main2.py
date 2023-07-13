import io
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, render_template

model = tf.keras.models.load_model("Best_pretrained.h5")

class_names = ['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon', 'hamburger', 'ice_cream', 'pizza', 'ramen', 'steak', 'sushi']

def transform_image(pillow_image):
    # Convert the image to RGB if it's grayscale
    if pillow_image.mode != "RGB":
        pillow_image = pillow_image.convert("RGB")
    
    # Resize the image to the desired input shape
    image_size = (224, 224)
    pillow_image = pillow_image.resize(image_size)

    # Convert the image to a numpy array
    image_array = np.array(pillow_image)

    # Normalize the pixel values to the range of [0, 1]
    # image_array = image_array / 255.0

    # Add an extra dimension to match the model's input shape
    data = np.expand_dims(image_array, axis=0)

    return data


def predict(x):
    predictions = model(x)
    pred0 = predictions[0]
    label0 = np.argmax(pred0)
    class_name = class_names[label0]
    return class_name

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_image():
    if request.method == "POST":
        file = request.files.get('image')
        if file is None or file.filename == "":
            return render_template("index.html", prediction_text="No file selected.")

        try:
            image_bytes = file.read()
            pillow_img = Image.open(io.BytesIO(image_bytes))
            tensor = transform_image(pillow_img)
            prediction = model.predict(tensor)
            pred_label = np.argmax(prediction)
            class_name = class_names[pred_label]
            image_path = os.path.join("static", file.filename)
            pillow_img.save(image_path)
            return render_template("index.html", prediction_text="The predicted image is {}".format(class_name), image_path=image_path)
        except Exception as e:
            return render_template("index.html", prediction_text="Error: {}".format(str(e)), image_path="")

    return "OK"

if __name__ == "__main__":
    app.run(debug=True)

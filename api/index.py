from flask import Flask, jsonify, request
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from io import BytesIO
import urllib.request as urllib

app = Flask(__name__)
classes = [
    "Bevel-gear",
    "bearing",
    "clutch",
    "cylincer",
    "filter",
    "fuel-tank",
    "helical_gear",
    "piston",
    "rack-pinion",
    "shocker",
    "spark-plug",
    "spur-gear",
    "valve",
    "wheel",
]


@app.route("/")
def home():
    return "Hello, World!"


@app.route("/predict", methods=["GET"])
def get_data():
    image_path = request.args.get("url")
    inception_resnet_trained_model = load_model("./classifier-model.h5")

    with urllib.urlopen(image_path) as url:
        img = load_img(BytesIO(url.read()), target_size=(224, 224))

    result = inception_resnet_trained_model.predict(
        img_to_array(img).reshape(1, 224, 224, 3)
    )

    classifier_predicted_class = np.argmax(result)
    print(classes[classifier_predicted_class])
    return jsonify({"class": classes[classifier_predicted_class]})


if __name__ == "__main__":
    app.run(debug=True)

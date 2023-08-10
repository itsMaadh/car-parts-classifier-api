from flask import Flask, jsonify, request
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from io import BytesIO
import urllib.request as urllib
import requests

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


def download_file(url, local_path):
    print(f"Started downloading file from url: ", url)
    response = requests.get(url)

    if response.status_code == 200:
        with open(local_path, "wb") as file:
            file.write(response.content)
            print(f"File downloaded and saved as {local_path}")
    else:
        print("Failed to download the file")


download_file(
    "https://lottiefiles-test.s3.ap-southeast-1.amazonaws.com/classifier-model.h5",
    "model.h5",
)


@app.route("/")
def home():
    return "Hello, World!"


@app.route("/predict", methods=["GET"])
def get_data():
    image_path = request.args.get("url")

    inception_resnet_trained_model = load_model("./model.h5")

    with urllib.urlopen(image_path) as url:
        img = load_img(BytesIO(url.read()), target_size=(224, 224))

    result = inception_resnet_trained_model.predict(
        img_to_array(img).reshape(1, 224, 224, 3)
    )

    classifier_predicted_class = np.argmax(result)
    return jsonify({"class": classes[classifier_predicted_class]})


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=2000)

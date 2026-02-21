import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

import gdown
import os

# ---------------------------------------------------------
# Model1
# ---------------------------------------------------------
model1_path = "models/model1.keras"
if not os.path.exists(model1_path):
    url1 = "https://drive.google.com/file/d/1JErxwqUtKiHxb4HFBkjN5aHiR4-JwP64/view?usp=drive_link"
    gdown.download(url1, model1_path, quiet=False)

# ---------------------------------------------------------
# Model2
# ---------------------------------------------------------
model2_path = "models/model2.h5"
if not os.path.exists(model2_path):
    url2 = "https://drive.google.com/file/d/1kRLRjbuFWyNh0UwBlnk0LUz59rblwahR/view?usp=drive_link"
    gdown.download(url2, model2_path, quiet=False)

# ---------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------
# MODEL 1: PLANT DISEASE MODEL
# ---------------------------------------------------------
model1 = tf.keras.models.load_model('models/model1.keras')

classes1 = [
    'Chrysanthemum_Bacterial_Leaf_Spot',
    'Chrysanthemum_Healthy',
    'Chrysanthemum_Septoria_Leaf_Spot',
    'Hibiscus_Blight',
    'Hibiscus_Healthy',
    'Hibiscus_Necrosis',
    'Hibiscus_Scorch',
    'Money Plant_Money_Plant_Bacterial_Wilt',
    'Money Plant_Money_Plant_Chlorosis',
    'Money Plant_Money_Plant_Healthy',
    'Money Plant_Money_Plant_Manganese_Toxicity',
    'Rose_Black_Spot',
    'Rose_Downy_Mildew',
    'Rose_Healthy',
    'Rose_Mosaic_Virus',
    'Rose_Powdery_Mildew',
    'Rose_Rust',
    'Rose_Yellow_Mosaic_Virus',
    'Turmeric_Aphid_Infestation',
    'Turmeric_Blotch',
    'Turmeric_Healthy',
    'Turmeric_Leaf_Necrosis',
    'Turmeric_Leaf_Spot'
]

# ---------------------------------------------------------
# MODEL 2: PLANT SPECIES MODEL
# ---------------------------------------------------------
model2 = tf.keras.models.load_model('models/model2.h5')

classes2 = [
    'African Violet (Saintpaulia ionantha)',
    'Aloe Vera',
    'Anthurium (Anthurium andraeanum)',
    'Areca Palm (Dypsis lutescens)',
    'Asparagus Fern (Asparagus setaceus)',
    'Begonia (Begonia spp.)',
    'Bird of Paradise (Strelitzia reginae)',
    'Birds Nest Fern (Asplenium nidus)',
    'Boston Fern (Nephrolepis exaltata)',
    'Calathea',
    'Cast Iron Plant (Aspidistra elatior)',
    'Chinese Money Plant (Pilea peperomioides)',
    'Chinese evergreen (Aglaonema)',
    'Christmas Cactus (Schlumbergera bridgesii)',
    'Chrysanthemum',
    'Ctenanthe',
    'Daffodils (Narcissus spp.)',
    'Dracaena',
    'Dumb Cane (Dieffenbachia spp.)',
    'Elephant Ear (Alocasia spp.)',
    'English Ivy (Hedera helix)',
    'Hyacinth (Hyacinthus orientalis)',
    'Iron Cross begonia (Begonia masoniana)',
    'Jade plant (Crassula ovata)',
    'Kalanchoe',
    'Lilium (Hemerocallis)',
    'Lily of the valley (Convallaria majalis)',
    'Money Tree (Pachira aquatica)',
    'Monstera Deliciosa (Monstera deliciosa)',
    'Orchid',
    'Parlor Palm (Chamaedorea elegans)',
    'Peace lily',
    'Poinsettia (Euphorbia pulcherrima)',
    'Polka Dot Plant (Hypoestes phyllostachya)',
    'Ponytail Palm (Beaucarnea recurvata)',
    'Pothos (Ivy arum)',
    'Prayer Plant (Maranta leuconeura)',
    'Rattlesnake Plant (Calathea lancifolia)',
    'Rubber Plant (Ficus elastica)',
    'Sago Palm (Cycas revoluta)',
    'Schefflera',
    'Snake plant (Sanseviera)',
    'Tradescantia',
    'Tulip',
    'Venus Flytrap',
    'Yucca',
    'ZZ Plant (Zamioculcas zamiifolia)'
]

# ---------------------------------------------------------
# SHARED IMAGE PREPROCESSING
# ---------------------------------------------------------
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    arr = tf.keras.preprocessing.image.img_to_array(img)  # (1, 224, 224, 3)
    arr = tf.expand_dims(arr, 0)
    return arr

# ---------------------------------------------------------
# MODEL 1 HELPERS
# ---------------------------------------------------------
def classify_model1(image):
    arr = preprocess_image(image)
    prediction = model1.predict(arr) # [0.001, 0.91, 0.0002, ]
    idx = np.argmax(prediction)  # 1
    return {
        "label": classes1[idx],
        "probability": float(np.max(prediction))
    }

# ---------------------------------------------------------
# MODEL 2 HELPERS (WITH GEMINI CARE TIPS)
# ---------------------------------------------------------
def classify_model2(image):
    arr = preprocess_image(image, target_size=(256, 256))
    prediction = model2.predict(arr)
    idx = np.argmax(prediction)
    return {
        "label": classes2[idx],
        "probability": float(np.max(prediction))
    }

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# MODEL 1: disease + Gemini treatment
@app.route("/api/classify", methods=["POST"])
def api_classify_model1():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image_file = request.files["image"]
        image = Image.open(image_file)
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    prediction = classify_model1(image)

    return jsonify({
        "success": True,
        "model": "plant_disease_classifier",
        "prediction": prediction
    })

@app.route("/classify", methods=["POST"])
def classify_web():
    if "image" not in request.files:
        return render_template("index.html", error="Please upload an image")

    image_file = request.files["image"]
    model_type = request.form.get("model")

    try:
        image = Image.open(image_file)
    except Exception:
        return render_template("index.html", error="Invalid image")

    if model_type == "disease":
        prediction = classify_model1(image)
        title = "Disease Detection"

    elif model_type == "species":
        prediction = classify_model2(image)
        title = "Plant Species"

    else:
        return render_template("index.html", error="Invalid model selection")

    return render_template(
        "result.html",
        title=title,
        label=prediction["label"],
        probability=round(prediction["probability"] * 100, 2)
    )


# MODEL 2: plant species classifier + Gemini care tips
@app.route("/api/classify2", methods=["POST"])
def api_classify_model2():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image_file = request.files["image"]
        image = Image.open(image_file)
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    prediction = classify_model2(image)

    return jsonify({
        "success": True,
        "model": "plant_species_classifier",
        "prediction": prediction
    })

# ---------------------------------------------------------
# RUN SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)

import os
import cv2
import numpy as np
from PIL import Image
import datetime
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

import tensorflow as tf
from keras.models import load_model

# PDF Generation Libraries
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet


app = Flask(__name__)


# ============================================================
# 🔥 LOAD TUMOR DETECTION MODEL
# ============================================================
print("Loading Brain Tumor Model...")
try:
    model = load_model('BrainTumor10EpochsCategorical.h5')
    print("Model Loaded Successfully")
except Exception as e:
    print("Error loading model:", e)
    model = None


# ============================================================
# 🔥 LOAD MODEL ACCURACY (Shown on Website)
# ============================================================
MODEL_ACCURACY = "N/A"
try:
    with open("model_accuracy.txt", "r") as f:
        MODEL_ACCURACY = f.read().strip()
except:
    print("model_accuracy.txt not found.")


# ============================================================
# 🔥 MRI VALIDATION (HSV Saturation Method)
# ============================================================
def validate_image(img_path):
    try:
        img = cv2.imread(img_path)

        if img is None:
            return False, "Unreadable Image"

        img_small = cv2.resize(img, (128, 128))

        hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        colorful_ratio = np.mean(saturation > 60)

        if colorful_ratio > 0.15:
            return False, "Too colorful — not MRI"

        return True, "Likely MRI"

    except Exception as e:
        print("MRI Validation Error:", e)
        return True, "Validation Skipped"


# ============================================================
# 🔥 TUMOR PREDICTION FUNCTION
# ============================================================
def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor Detected"
    elif classNo == 1:
        return "Yes, Brain Tumor Detected"
    return "Unknown"


def getResult(img_path):
    img_cv = cv2.imread(img_path)

    if img_cv is None:
        return None

    img_pil = Image.fromarray(img_cv, 'RGB').resize((64, 64))
    img_np = np.array(img_pil)

    if img_np.ndim == 2 or img_np.shape[-1] == 1:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[-1] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

    img_np = img_np / 255.0
    img_expanded = np.expand_dims(img_np, axis=0)

    prediction = model.predict(img_expanded)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    return class_idx, confidence


# ============================================================
# 🔥 PDF REPORT GENERATOR
# ============================================================
def generate_report(image_path, prediction, confidence, patient_name="Patient"):
    try:
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_folder = "static/reports"
        if not os.path.exists(report_folder):
            os.makedirs(report_folder)

        report_filename = f"{report_folder}/Report_{date_str}.pdf"

        doc = SimpleDocTemplate(report_filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("<b><font size=18>Brain Tumor Analysis Report</font></b>", styles['Title']))
        story.append(Spacer(1, 18))

        # Info Section
        info = Paragraph(
            f"""
            <b>Patient Name:</b> {patient_name}<br/>
            <b>Date:</b> {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}<br/>
            <b>Prediction:</b> {prediction}<br/>
            <b>Confidence:</b> {confidence}%<br/>
            """, styles['Normal']
        )
        story.append(info)
        story.append(Spacer(1, 20))

        # MRI Image
        story.append(Paragraph("<b>MRI Image Used:</b>", styles['Heading3']))
        story.append(Spacer(1, 10))

        try:
            img = RLImage(image_path, width=300, height=300)
            story.append(img)
        except:
            story.append(Paragraph("Unable to load MRI image.", styles['Normal']))

        story.append(Spacer(1, 25))

        # Footer
        story.append(Paragraph(
            "<i>This report is AI-generated and should not replace professional medical diagnosis.</i>",
            styles['Italic']
        ))

        doc.build(story)

        return report_filename

    except Exception as e:
        print("PDF Report Error:", e)
        return None


# ============================================================
# 🔥 FLASK ROUTES
# ============================================================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", accuracy=MODEL_ACCURACY)


@app.route("/predict", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'})

    # NEW: Receive patient name
    patient_name = request.form.get("patient_name", "Patient")

    # Save uploaded file
    basepath = os.path.dirname(__file__)
    upload_folder = os.path.join(basepath, 'static', 'uploads')
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    file_path = os.path.join(upload_folder, secure_filename(f.filename))
    f.save(file_path)

    print("Image received:", file_path)

    # 🔍 Step 1: Validate MRI
    is_mri, msg = validate_image(file_path)
    if not is_mri:
        return jsonify({'error': 'Please Upload Brain MRI image'})

    # 🧠 Step 2: Predict Tumor
    if model is None:
        return jsonify({'error': 'Model not loaded'})

    output = getResult(file_path)
    if output is None:
        return jsonify({'error': 'Image processing failed'})

    class_no, confidence = output
    prediction_text = get_className(class_no)

    # 📄 Step 3: Generate PDF Report
    report_path = generate_report(
        file_path,
        prediction_text,
        f"{confidence:.2f}",
        patient_name=patient_name
    )

    return jsonify({
        'prediction': prediction_text,
        'confidence': f"{confidence:.2f}",
        'report': report_path
    })


# ============================================================
# RUN FLASK APP
# ============================================================
if __name__ == "__main__":
    print("Server running at http://127.0.0.1:5000/")
    app.run(debug=True)

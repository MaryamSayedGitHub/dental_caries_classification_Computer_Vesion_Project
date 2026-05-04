from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
from ultralytics import YOLO
import supervision as sv
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
# Ensure this matches your folder name: "outputs"
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "outputs") 
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# Load YOLO model
model = YOLO("last.pt")

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def run_detection(image_path, filename):
    img = cv2.imread(image_path)
    if img is None:
        return None, [], 0

    # 1. Run Inference
    results = model(img)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # 2. Detailed Analysis: Extract labels and count cavities
    raw_labels = [model.names[class_id] for class_id in detections.class_id]
    cavity_count = raw_labels.count('cavity')
    
    # Create descriptive labels for the image (e.g., "cavity 0.85")
    display_labels = [
        f"{model.names[class_id]} {conf:.2f}" 
        for class_id, conf in zip(detections.class_id, detections.confidence)
    ]

    # 3. Annotation
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, 
        detections=detections, 
        labels=display_labels
    )

    # 4. Save to static/outputs/
    output_path = os.path.join(app.config["RESULT_FOLDER"], filename)
    cv2.imwrite(output_path, annotated_frame)
    
    return filename, list(set(raw_labels)), cavity_count

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(saved_path)

    # Process and get detailed results
    result_filename, detected_classes, cavity_count = run_detection(saved_path, filename)

    if result_filename:
        return render_template(
            "results.html", 
            result_paths=[result_filename], 
            detected_classes=detected_classes,
            cavity_count=cavity_count
        )
    
    return "Error processing image."

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
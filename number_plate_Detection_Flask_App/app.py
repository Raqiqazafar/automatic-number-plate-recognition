from flask import Flask, render_template, request
import os
import cv2
from ultralytics import YOLO
import easyocr
import matplotlib.pyplot as plt

# ---------------- Paths
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ---------------- Flask app
app = Flask(__name__)

# ---------------- Load YOLO + OCR
model = YOLO("best_license_plate_model.pt")  # Make sure file is in project folder
reader = easyocr.Reader(['en'], gpu=True)

# ---------------- Detection + OCR
def detect_and_read(img_path):
    results = model.predict(img_path, imgsz=640, conf=0.55, save=False, verbose=False)
    
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_text = ""

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > 0.55:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = image[y1:y2, x1:x2]

                ocr_result = reader.readtext(plate_crop)
                if len(ocr_result) > 0:
                    detected_text = " ".join([res[1] for res in ocr_result])

                cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(image, detected_text, (x1, max(y1-10,0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    
    # Save result
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(img_path))
    plt.imsave(result_path, image)
    return result_path, detected_text

# ---------------- Routes
@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(upload_path)

            result_path, detected_text = detect_and_read(upload_path)

            return render_template("index.html",
                                   original=upload_path,
                                   result=result_path,
                                   detected_text=detected_text)
    return render_template("index.html", original=None, result=None, detected_text=None)

# ---------------- Run server
if __name__ == "__main__":
    app.run(debug=True)

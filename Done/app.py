from flask import render_template
from flask import Flask, render_template, request, Response
import os
from werkzeug.utils import secure_filename
from yolo_detect import detect_from_image, detect_from_video, generate_frames

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return "No file part"
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    filename = secure_filename(file.filename)
    ext = filename.split('.')[-1].lower()
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    if ext in ['jpg', 'jpeg', 'png']:
        result_filename, person_count = detect_from_image(file_path)
        return render_template('result.html', filename=result_filename, person_count=person_count)
    
    elif ext in ['mp4', 'avi', 'mov']:
        person_count = detect_from_video(file_path)
        return render_template('result.html', filename=None, person_count=person_count)
    
    return "Unsupported file format"

@app.route('/live')
def live():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

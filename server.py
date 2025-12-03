# server.py - Flask server integrating KNN face recognition
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import face_recognition
import numpy as np
import cv2
import os
import pickle
import base64
from PIL import Image
import io
import json
from datetime import datetime
import math
from sklearn import neighbors
from face_recognition.face_recognition_cli import image_files_in_folder

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_FILE = "trained_knn_model.clf"
TRAIN_DIR = "training_images"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create directories if they don't exist
os.makedirs(TRAIN_DIR, exist_ok=True)


class KNNFaceRecognizer:
    def __init__(self):
        self.model_file = MODEL_FILE
        self.train_dir = TRAIN_DIR
        self.knn_clf = None
        self.load_model()

    def load_model(self):
        """Load existing KNN model if available"""
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    self.knn_clf = pickle.load(f)
                print(f"Loaded existing KNN model from {self.model_file}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.knn_clf = None
        else:
            print("No existing model found. Will train when faces are added.")

    def train_model(self, n_neighbors=None, knn_algo='ball_tree', verbose=True):
        """Train KNN classifier on current training images"""
        X = []
        y = []

        if not os.path.exists(self.train_dir):
            return False, "Training directory not found"

        person_dirs = [d for d in os.listdir(self.train_dir)
                       if os.path.isdir(os.path.join(self.train_dir, d))]

        if len(person_dirs) == 0:
            return False, "No training data found"

        print(f"Training on {len(person_dirs)} people...")

        # Loop through each person in the training set
        for class_dir in person_dirs:
            person_path = os.path.join(self.train_dir, class_dir)

            # Loop through each training image for the current person
            image_files = [f for f in os.listdir(person_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for img_file in image_files:
                img_path = os.path.join(person_path, img_file)
                try:
                    image = face_recognition.load_image_file(img_path)
                    face_bounding_boxes = face_recognition.face_locations(image)

                    if len(face_bounding_boxes) != 1:
                        if verbose:
                            print(
                                f"Skipping {img_path}: {'No face found' if len(face_bounding_boxes) < 1 else 'Multiple faces found'}")
                        continue
                    else:
                        # Add face encoding for current image to the training set
                        face_encoding = \
                        face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
                        X.append(face_encoding)
                        y.append(class_dir)

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

        if len(X) == 0:
            return False, "No valid training faces found"

        # Determine how many neighbors to use for weighting in the KNN classifier
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            if verbose:
                print(f"Chose n_neighbors automatically: {n_neighbors}")

        # Create and train the KNN classifier
        self.knn_clf = neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors,
            algorithm=knn_algo,
            weights='distance'
        )
        self.knn_clf.fit(X, y)

        # Save the trained KNN classifier
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.knn_clf, f)
            print(f"Model saved to {self.model_file}")
        except Exception as e:
            print(f"Error saving model: {e}")

        return True, f"Successfully trained on {len(X)} face encodings from {len(set(y))} people"

    def predict(self, frame, distance_threshold=0.5):
        """Predict faces in frame using KNN model"""
        if self.knn_clf is None:
            return []

        face_locations = face_recognition.face_locations(frame)

        if len(face_locations) == 0:
            return []

        # Find encodings for faces in the test image
        faces_encodings = face_recognition.face_encodings(frame, known_face_locations=face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        predictions = []
        predicted_names = self.knn_clf.predict(faces_encodings)

        for pred, loc, rec, dist in zip(predicted_names, face_locations, are_matches, closest_distances[0]):
            name = pred if rec else "Unknown"
            confidence = 1 - dist[0] if rec else 0
            predictions.append((name, loc, confidence))

        return predictions

    def get_known_people(self):
        """Get list of people in training directory"""
        if not os.path.exists(self.train_dir):
            return []

        return [d for d in os.listdir(self.train_dir)
                if os.path.isdir(os.path.join(self.train_dir, d))]


# Initialize face recognizer
face_recognizer = KNNFaceRecognizer()


@app.route('/')
def mobile_app():
    """Serve the mobile app"""
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KNN Face Recognition - Add Person</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            min-height: 100vh;
            padding: 40px 20px;
        }

        .container {
            max-width: 480px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.08);
        }

        .header {
            margin-bottom: 32px;
        }

        .header h1 {
            color: #1a202c;
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 6px;
            letter-spacing: -0.02em;
        }

        .header p {
            color: #718096;
            font-size: 14px;
            font-weight: 400;
        }

        .camera-section {
            margin-bottom: 24px;
        }

        #video {
            width: 100%;
            height: 280px;
            border-radius: 12px;
            object-fit: cover;
            background: #e2e8f0;
            border: 1px solid #e2e8f0;
        }

        .camera-controls {
            display: flex;
            gap: 12px;
            margin-top: 16px;
        }

        .btn {
            flex: 1;
            padding: 12px 16px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }

        .btn-primary {
            background: #2563eb;
            color: white;
        }

        .btn-primary:hover:not(:disabled) {
            background: #1d4ed8;
        }

        .btn-success {
            background: #10b981;
            color: white;
        }

        .btn-success:hover:not(:disabled) {
            background: #059669;
        }

        .btn-danger {
            background: #ef4444;
            color: white;
        }

        .btn-warning {
            background: #f59e0b;
            color: white;
        }

        .btn-warning:hover:not(:disabled) {
            background: #d97706;
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .preview-section {
            margin-bottom: 24px;
            display: none;
        }

        .preview-section h3 {
            color: #1a202c;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 12px;
        }

        #preview {
            width: 100%;
            max-height: 240px;
            border-radius: 12px;
            object-fit: cover;
            border: 1px solid #e2e8f0;
        }

        .input-group {
            margin-bottom: 24px;
        }

        .input-group label {
            display: block;
            margin-bottom: 8px;
            color: #374151;
            font-weight: 600;
            font-size: 14px;
        }

        .input-group input {
            width: 100%;
            padding: 12px 14px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            font-size: 14px;
            transition: all 0.2s ease;
            background: white;
        }

        .input-group input:focus {
            outline: none;
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }

        .status {
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
            font-size: 14px;
            font-weight: 500;
        }

        .status.success {
            background: #d1fae5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }

        .status.error {
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }

        .train-section {
            margin-top: 24px;
            padding: 16px;
            background: #fef3c7;
            border-radius: 10px;
            border: 1px solid #fde68a;
        }

        .train-section p {
            margin-top: 8px;
            font-size: 13px;
            color: #78350f;
        }

        .divider {
            height: 1px;
            background: #e5e7eb;
            margin: 32px 0;
        }

        .known-faces h3 {
            color: #1a202c;
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .badge {
            background: #e5e7eb;
            color: #374151;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }

        .face-list {
            max-height: 200px;
            overflow-y: auto;
        }

        .face-list::-webkit-scrollbar {
            width: 6px;
        }

        .face-list::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 3px;
        }

        .face-list::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 3px;
        }

        .face-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 14px;
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            margin-bottom: 8px;
            font-size: 14px;
        }

        .face-item:last-child {
            margin-bottom: 0;
        }

        .face-item span {
            color: #1a202c;
            font-weight: 500;
        }

        .delete-btn {
            background: #ef4444;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .delete-btn:hover {
            background: #dc2626;
        }

        .empty-state {
            text-align: center;
            padding: 32px 16px;
            color: #9ca3af;
            font-size: 14px;
        }

        @media (max-width: 480px) {
            .container {
                padding: 24px;
            }

            .camera-controls {
                flex-direction: column;
            }

            body {
                padding: 20px 16px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Face Recognition System</h1>
            <p>Add people to your recognition database</p>
        </div>

        <div class="camera-section">
            <video id="video" autoplay playsinline></video>
            <div class="camera-controls">
                <button id="startCamera" class="btn btn-primary">
                    <span>📷</span>
                    <span>Start Camera</span>
                </button>
                <button id="capturePhoto" class="btn btn-success" disabled>
                    <span>📸</span>
                    <span>Capture</span>
                </button>
            </div>
        </div>

        <div class="preview-section" id="previewSection">
            <h3>Preview</h3>
            <img id="preview" alt="Captured photo">
        </div>

        <div id="status" class="status"></div>

        <div class="input-group">
            <label for="personName">Person's Name</label>
            <input type="text" id="personName" placeholder="Enter name" required>
        </div>

        <button id="submitFace" class="btn btn-primary" style="width: 100%;" disabled>Add Person</button>

        <div class="train-section">
            <button id="trainModel" class="btn btn-warning" style="width: 100%;">Train Model</button>
            <p>Train the model after adding new people to enable recognition</p>
        </div>

        <div class="divider"></div>

        <div class="known-faces">
            <h3>
                Known People
                <span class="badge" id="faceCount">0</span>
            </h3>
            <div id="faceList" class="face-list"></div>
        </div>
    </div>

    <script>
        const video = document.getElementById('video');
        const startCameraBtn = document.getElementById('startCamera');
        const captureBtn = document.getElementById('capturePhoto');
        const preview = document.getElementById('preview');
        const previewSection = document.getElementById('previewSection');
        const personNameInput = document.getElementById('personName');
        const submitBtn = document.getElementById('submitFace');
        const trainBtn = document.getElementById('trainModel');
        const status = document.getElementById('status');
        const faceList = document.getElementById('faceList');
        const faceCount = document.getElementById('faceCount');

        let stream = null;
        let capturedImageData = null;

        const SERVER_URL = window.location.origin;

        startCameraBtn.addEventListener('click', async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        facingMode: 'user',
                        width: { ideal: 640 },
                        height: { ideal: 480 }
                    } 
                });
                video.srcObject = stream;
                startCameraBtn.disabled = false;
                captureBtn.disabled = false;
                showStatus('Camera started successfully', 'success');
            } catch (err) {
                showStatus('Error accessing camera: ' + err.message, 'error');
            }
        });

        captureBtn.addEventListener('click', () => {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);

            capturedImageData = canvas.toDataURL('image/jpeg', 0.8);
            preview.src = capturedImageData;
            previewSection.style.display = 'block';

            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            video.srcObject = null;

            startCameraBtn.disabled = false;
            captureBtn.disabled = true;
            submitBtn.disabled = false;

            showStatus('Photo captured successfully', 'success');
        });

        submitBtn.addEventListener('click', async () => {
            const name = personNameInput.value.trim();

            if (!name) {
                showStatus('Please enter a name', 'error');
                return;
            }

            if (!capturedImageData) {
                showStatus('Please capture a photo first', 'error');
                return;
            }

            submitBtn.disabled = true;
            showStatus('Adding person to database...', 'success');

            try {
                const response = await fetch('/api/add_face', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        name: name,
                        image: capturedImageData
                    })
                });

                const result = await response.json();

                if (response.ok) {
                    showStatus(`${result.message}`, 'success');
                    personNameInput.value = '';
                    previewSection.style.display = 'none';
                    capturedImageData = null;
                    loadKnownPeople();
                } else {
                    showStatus(`${result.error}`, 'error');
                }
            } catch (err) {
                showStatus(`Network error: ${err.message}`, 'error');
            }

            submitBtn.disabled = false;
        });

        trainBtn.addEventListener('click', async () => {
            trainBtn.disabled = true;
            showStatus('Training model... Please wait', 'success');

            try {
                const response = await fetch('/api/train_model', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });

                const result = await response.json();

                if (response.ok) {
                    showStatus(`${result.message}`, 'success');
                } else {
                    showStatus(`${result.error}`, 'error');
                }
            } catch (err) {
                showStatus(`Network error: ${err.message}`, 'error');
            }

            trainBtn.disabled = false;
        });

        async function loadKnownPeople() {
            try {
                const response = await fetch('/api/list_people');
                const result = await response.json();

                faceCount.textContent = result.total_count;

                faceList.innerHTML = '';
                
                if (result.people.length === 0) {
                    faceList.innerHTML = '<div class="empty-state">No people added yet</div>';
                } else {
                    result.people.forEach(person => {
                        const faceItem = document.createElement('div');
                        faceItem.className = 'face-item';
                        faceItem.innerHTML = `
                            <span>${person}</span>
                            <button class="delete-btn" onclick="deletePerson('${person}')">Delete</button>
                        `;
                        faceList.appendChild(faceItem);
                    });
                }
            } catch (err) {
                console.error('Failed to load people:', err);
            }
        }

        async function deletePerson(name) {
            if (!confirm(`Delete ${name} from the database?`)) {
                return;
            }

            try {
                const response = await fetch('/api/delete_person', {
                    method: 'DELETE',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ name: name })
                });

                const result = await response.json();

                if (response.ok) {
                    showStatus(`${result.message}`, 'success');
                    loadKnownPeople();
                } else {
                    showStatus(`${result.error}`, 'error');
                }
            } catch (err) {
                showStatus(`Network error: ${err.message}`, 'error');
            }
        }

        window.deletePerson = deletePerson;

        function showStatus(message, type) {
            status.textContent = message;
            status.className = `status ${type}`;
            status.style.display = 'block';

            if (type === 'success') {
                setTimeout(() => {
                    status.style.display = 'none';
                }, 4000);
            }
        }

        loadKnownPeople();
    </script>
</body>
</html>
''')


@app.route('/api/add_face', methods=['POST'])
def add_face():
    """Add a new face to the training set"""
    try:
        data = request.get_json()

        if 'image' not in data or 'name' not in data:
            return jsonify({'error': 'Missing image or name'}), 400

        name = data['name'].strip()
        if not name:
            return jsonify({'error': 'Name cannot be empty'}), 400

        # Create person directory
        person_dir = os.path.join(TRAIN_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'].split(',')[1])
            image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400

        # Verify face exists in image
        image_array = np.array(image)
        face_locations = face_recognition.face_locations(image_array)

        if len(face_locations) == 0:
            return jsonify({'error': 'No face found in image'}), 400
        elif len(face_locations) > 1:
            return jsonify({'error': 'Multiple faces found. Please use an image with only one face'}), 400

        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.jpg"
        filepath = os.path.join(person_dir, filename)
        image.save(filepath, 'JPEG', quality=95)

        # Count images for this person
        image_count = len([f for f in os.listdir(person_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        return jsonify({
            'message': f'Successfully added image for {name} (total: {image_count} images)',
            'recommendation': 'Add 3-5 more images of this person for better recognition' if image_count < 4 else 'Good! You can now train the model.'
        })

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/list_people', methods=['GET'])
def list_people():
    """Get list of all people in training set"""
    people = face_recognizer.get_known_people()
    return jsonify({
        'people': people,
        'total_count': len(people)
    })


@app.route('/api/delete_person', methods=['DELETE'])
def delete_person():
    """Delete a person from the training set"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()

        if not name:
            return jsonify({'error': 'Name is required'}), 400

        person_dir = os.path.join(TRAIN_DIR, name)

        if not os.path.exists(person_dir):
            return jsonify({'error': f'Person "{name}" not found'}), 404

        # Remove all images for this person
        import shutil
        shutil.rmtree(person_dir)

        return jsonify({'message': f'Successfully deleted all data for {name}. Remember to retrain the model!'})

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/train_model', methods=['POST'])
def train_model():
    """Train the KNN model on current training data"""
    try:
        success, message = face_recognizer.train_model()

        if success:
            return jsonify({'message': message})
        else:
            return jsonify({'error': message}), 400

    except Exception as e:
        return jsonify({'error': f'Training error: {str(e)}'}), 500


@app.route('/api/predict', methods=['POST'])
def predict_faces():
    """Predict faces in uploaded image"""
    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'error': 'Missing image'}), 400

        # Decode image
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        frame = np.array(image)

        # Convert BGR to RGB if necessary
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get predictions
        predictions = face_recognizer.predict(frame)

        # Format results
        results = []
        for name, location, confidence in predictions:
            top, right, bottom, left = location
            results.append({
                'name': name,
                'confidence': float(confidence),
                'location': {
                    'top': int(top),
                    'right': int(right),
                    'bottom': int(bottom),
                    'left': int(left)
                }
            })

        return jsonify({
            'faces': results,
            'count': len(results)
        })

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


def show_prediction_labels_on_image(frame, predictions):
    """Shows the face recognition results visually (updated for KNN)"""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left), confidence in predictions:
        # Choose color based on recognition
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, red for unknown

        # Draw rectangle around face
        draw.rectangle(((left, top), (right, bottom)), outline=color, width=3)

        # Prepare label text with confidence
        if name != "Unknown":
            label = f"{name} ({confidence:.2f})"
        else:
            label = name

        # Draw label
        bbox = draw.textbbox((0, 0), text=label)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw background for text
        draw.rectangle(((left, bottom - text_height - 10), (left + text_width + 12, bottom)),
                       fill=color, outline=color)

        # Draw text
        draw.text((left + 6, bottom - text_height - 5), label, fill=(255, 255, 255))

    del draw
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    print("KNN Face Recognition Server starting...")
    print("Server will run on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

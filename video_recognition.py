# video_recognition.py - KNN-based real-time face recognition
import cv2
import face_recognition
import numpy as np
import pickle
import os
import math
from sklearn import neighbors
from PIL import Image, ImageDraw


class KNNVideoFaceRecognition:
    def __init__(self):
        self.model_file = "trained_knn_model.clf"
        self.knn_clf = None
        self.load_model()

    def load_model(self):
        """Load the trained KNN model"""
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    self.knn_clf = pickle.load(f)
                print(f"✅ Loaded KNN model from {self.model_file}")
                return True
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                self.knn_clf = None
                return False
        else:
            print(f"❌ No model file found at {self.model_file}")
            print("Please train the model first using the mobile app!")
            return False

    def predict_faces(self, frame, distance_threshold=0.5):
        """Predict faces in frame using KNN model"""
        if self.knn_clf is None:
            return []

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find face locations
        face_locations = face_recognition.face_locations(rgb_small_frame)

        if len(face_locations) == 0:
            return []

        # Find encodings for faces in the frame
        faces_encodings = face_recognition.face_encodings(rgb_small_frame, known_face_locations=face_locations)

        predictions = []

        # Use the KNN model to find the best matches
        try:
            closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            predicted_names = self.knn_clf.predict(faces_encodings)

            for i, (face_encoding, face_location) in enumerate(zip(faces_encodings, face_locations)):
                # Check if the closest match is within threshold
                distance = closest_distances[0][i][0]
                is_match = distance <= distance_threshold

                name = predicted_names[i] if is_match else "Unknown"
                confidence = 1 - distance if is_match else 0

                # Scale back up face locations (we processed on smaller frame)
                top, right, bottom, left = face_location
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                predictions.append((name, (top, right, bottom, left), confidence))

        except Exception as e:
            print(f"Prediction error: {e}")
            return []

        return predictions

    def show_prediction_labels_on_image(self, frame, predictions):
        """Shows the face recognition results visually"""
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        for name, (top, right, bottom, left), confidence in predictions:
            # Choose color based on recognition
            color = (0, 0, 0) if name != "Unknown" else (255, 0, 0)  # Green for known, red for unknown

            # Draw rectangle around face
            draw.rectangle(((left, top), (right, bottom)), outline=color, width=3)

            # Prepare label text with confidence
            if name != "Unknown":
                label = f"{name} ({confidence:.2f})"
            else:
                label = "Unknown"

            # Calculate text size and draw background
            bbox = draw.textbbox((0, 0), text=label)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Draw background rectangle for text
            draw.rectangle(((left, bottom - text_height - 10), (left + text_width + 12, bottom)),
                           fill=color, outline=color)

            # Draw the text
            draw.text((left + 6, bottom - text_height - 5), label, fill=(255, 255, 255))

        del draw
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def run_video_recognition(self, camera_source=0, process_every_n_frames=5):
        """Run real-time face recognition on video stream"""

        if not self.load_model():
            print("Cannot start video recognition without a trained model.")
            print("Please use the mobile app to add people and train the model first!")
            return

        # Initialize video capture
        if isinstance(camera_source, str) and camera_source.startswith('http'):
            # IP camera
            cap = cv2.VideoCapture(camera_source)
        else:
            # Webcam
            cap = cv2.VideoCapture(camera_source)

        if not cap.isOpened():
            print(f"❌ Error: Could not open camera source: {camera_source}")
            return

        print("🎥 KNN Face Recognition System Started!")
        print("📋 Controls:")
        print("   'q' - Quit")
        print("   'r' - Reload model")
        print("   's' - Save current frame")
        print("   '+' - Increase confidence threshold")
        print("   '-' - Decrease confidence threshold")

        # Configuration
        frame_count = 0
        distance_threshold = 0.5
        predictions = []  # Store last predictions to avoid flickering

        print(f"🎯 Current distance threshold: {distance_threshold}")

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break

                frame_count += 1

                # Process face recognition every N frames for performance
                if frame_count % process_every_n_frames == 0:
                    predictions = self.predict_faces(frame, distance_threshold)

                # Draw predictions on frame
                if predictions:
                    result_frame = self.show_prediction_labels_on_image(frame, predictions)
                else:
                    result_frame = frame

                # Add info overlay
                info_text = f"Threshold: {distance_threshold:.2f} | Faces: {len(predictions)}"
                cv2.putText(result_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Display the frame
                cv2.imshow('KNN Face Recognition System', result_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("👋 Shutting down...")
                    break

                elif key == ord('r'):
                    print("🔄 Reloading model...")
                    if self.load_model():
                        print("✅ Model reloaded successfully!")
                    else:
                        print("❌ Failed to reload model")

                elif key == ord('s'):
                    # Save current frame
                    timestamp = cv2.getTickCount()
                    filename = f"captured_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, result_frame)
                    print(f"💾 Frame saved as {filename}")

                elif key == ord('+') or key == ord('='):
                    distance_threshold = min(1.0, distance_threshold + 0.05)
                    print(f"🎯 Distance threshold increased to: {distance_threshold:.2f}")

                elif key == ord('-'):
                    distance_threshold = max(0.1, distance_threshold - 0.05)
                    print(f"🎯 Distance threshold decreased to: {distance_threshold:.2f}")

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")

        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("🧹 Cleaned up resources")


def main():
    """Main function to run the face recognition system"""

    print("=" * 60)
    print("🤖 KNN Face Recognition System")
    print("=" * 60)

    # Initialize the recognition system
    recognizer = KNNVideoFaceRecognition()

    # Check if model exists
    if not os.path.exists("trained_knn_model.clf"):
        print("\n⚠️  No trained model found!")
        print("📱 Please use the mobile app to:")
        print("   1. Add photos of people you want to recognize")
        print("   2. Train the KNN model")
        print("   3. Then run this script again")
        print(f"\n🌐 Mobile app URL: http://localhost:5000")
        return

    # Configuration options
    print("\n⚙️  Configuration Options:")
    print("1. Use default webcam (recommended)")
    print("2. Use IP camera")
    print("3. Use different camera index")

    choice = input("\nEnter your choice (1-3) [default: 1]: ").strip()

    if choice == "2":
        camera_url = input("Enter IP camera URL (e.g., http://192.168.1.100:8090/video): ")
        camera_source = camera_url
    elif choice == "3":
        try:
            camera_index = int(input("Enter camera index (0, 1, 2, etc.): "))
            camera_source = camera_index
        except ValueError:
            print("Invalid input, using default camera (0)")
            camera_source = 0
    else:
        camera_source = 0  # Default webcam

    # Performance settings
    print("\n🚀 Performance Settings:")
    print("1. High quality (process every frame)")
    print("2. Balanced (process every 3 frames) - recommended")
    print("3. High speed (process every 5 frames)")

    perf_choice = input("\nEnter your choice (1-3) [default: 2]: ").strip()

    if perf_choice == "1":
        process_every = 1
    elif perf_choice == "3":
        process_every = 5
    else:
        process_every = 3  # Default

    print(f"\n🎬 Starting video recognition...")
    print(f"📷 Camera source: {camera_source}")
    print(f"⚡ Processing every {process_every} frames")

    # Start the recognition system
    recognizer.run_video_recognition(camera_source, process_every)


if __name__ == "__main__":
    main()
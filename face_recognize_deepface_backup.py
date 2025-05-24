# facerec.py
import cv2, os
import numpy as np
from deepface import DeepFace
import json
import time
from numpy.linalg import norm
import threading
from queue import Queue
import copy

# Global configuration variables
embeddings_dir = 'embeddings'  # Directory containing face embeddings
webcam_in_use = False
PROCESS_EVERY_N_FRAMES = 10  # Process every 10th frame instead of 30 for faster response
MIN_SIMILARITY = 70  # Slightly lower minimum similarity for faster recognition
MAX_FACE_AGE = 10  # Maximum age for a face to be considered active

# Global variables for threading
frame_queue = Queue(maxsize=1)  # Queue to pass frames between threads
recognition_results = {}  # Dictionary to store recognition results
recognition_lock = threading.Lock()  # Lock for thread-safe access to recognition_results
processing_active = True  # Flag to control worker thread

def normalize_embedding(embedding):
    """Normalize embedding vector to unit length"""
    return embedding / np.linalg.norm(embedding)

def calculate_similarity_percentage(distance):
    """
    Convert distance to similarity percentage
    Distance of 0 = 100% similarity
    Distance of 1 = 0% similarity
    """
    return max(0, min(100, (1 - distance) * 100))

def load_embeddings():
    """
    Load all face embeddings from the embeddings directory
    Returns:
        dict: Dictionary mapping IC numbers to their embeddings
    """
    embeddings = {}
    if not os.path.exists(embeddings_dir):
        print("❌ No embeddings directory found!")
        return embeddings

    for filename in os.listdir(embeddings_dir):
        if filename.endswith('_embedding.json'):
            ic = filename.replace('_embedding.json', '')
            with open(os.path.join(embeddings_dir, filename), 'r') as f:
                embedding = np.array(json.load(f))
                # Normalize the embedding
                embeddings[ic] = normalize_embedding(embedding)
    
    print(f"✅ Loaded {len(embeddings)} face embeddings")
    return embeddings

def process_frame(frame, known_embeddings, current_time):
    """
    Process a single frame for face recognition
    This function runs in a separate thread
    """
    try:
        # Detect faces using DeepFace
        face_objs = DeepFace.extract_faces(
            img_path=frame,
            detector_backend='retinaface',
            enforce_detection=True,
            align=True,
            anti_spoofing=False
        )
        print(f"\nDetected {len(face_objs)} face(s)")

        # Clear old results at the start of new detection
        with recognition_lock:
            recognition_results.clear()

        # Process each detected face
        for face_idx, face in enumerate(face_objs):
            facial_area = face.get('facial_area', {})
            x = facial_area.get('x', 0)
            y = facial_area.get('y', 0)
            w = facial_area.get('w', 0)
            h = facial_area.get('h', 0)

            try:
                # Extract face region with padding
                padding = int(min(w, h) * 0.2)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size == 0:  # Skip if face region is empty
                    continue
                
                face_img = cv2.resize(face_img, (224, 224))

                # Get face embedding
                reps = DeepFace.represent(
                    img_path=face_img,
                    model_name='ArcFace',
                    detector_backend='retinaface',
                    enforce_detection=True,
                    align=True,
                    normalization='base'
                )
                
                if not reps:  # Skip if no embeddings were generated
                    continue
                    
                current_embedding = normalize_embedding(np.array(reps[0]['embedding']))

                # Compare with known embeddings
                best_match = None
                best_distance = float('inf')
                best_similarity = 0

                print(f"\nProcessing Face {face_idx + 1}:")
                print(f"Comparing with {len(known_embeddings)} stored embeddings...")
                
                for ic, known_embedding in known_embeddings.items():
                    distance = 1 - np.dot(current_embedding, known_embedding)
                    similarity = calculate_similarity_percentage(distance)
                    print(f"🔍 IC: {ic} - Distance: {distance:.4f}, Similarity: {similarity:.2f}%")
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = ic
                        best_similarity = similarity

                # Print recognition result
                print(f"\n{'✅' if best_similarity >= MIN_SIMILARITY else '❌'} Face {face_idx + 1} {'Recognized!' if best_similarity >= MIN_SIMILARITY else 'Unknown'}")
                if best_similarity >= MIN_SIMILARITY:
                    print(f"IC: {best_match}")
                    print(f"Similarity: {best_similarity:.2f}%")
                    print(f"Distance: {best_distance:.4f}")
                else:
                    print(f"Best Match Similarity: {best_similarity:.2f}%")
                    print(f"Distance: {best_distance:.4f}")
                    print(f"Minimum Required Similarity: {MIN_SIMILARITY}%")
                print("-" * 30)

                face_key = f"{x}_{y}_{w}_{h}"
                if face_key not in recognition_results:
                    # New face detected
                    recognition_results[face_key] = {
                        'last_recognized': best_match if best_similarity >= MIN_SIMILARITY else "Unknown",
                        'bbox': (x, y, w, h),
                        'similarity': best_similarity
                    }

                # Update face information
                recognition_results[face_key].update({
                    'last_recognized': best_match if best_similarity >= MIN_SIMILARITY else "Unknown",
                    'bbox': (x, y, w, h),
                    'similarity': best_similarity
                })

            except Exception as e:
                print(f"\n❌ [Recognition Error for Face {face_idx + 1}] {str(e)}")
                face_key = f"{x}_{y}_{w}_{h}"
                recognition_results[face_key] = {
                    'last_recognized': "Error",
                    'bbox': (x, y, w, h),
                    'similarity': 0
                }

    except Exception as e:
        print(f"\n❌ [Detection Error] {str(e)}")

def recognition_worker(known_embeddings):
    """
    Worker thread function that processes frames from the queue
    """
    global processing_active
    print("Recognition worker thread started")
    
    while processing_active:
        try:
            if not frame_queue.empty():
                frame, current_time = frame_queue.get()
                process_frame(frame, known_embeddings, current_time)
                frame_queue.task_done()
            else:
                time.sleep(0.01)  # Small sleep when queue is empty
        except Exception as e:
            print(f"Worker thread error: {str(e)}")
            time.sleep(0.1)  # Sleep longer on error

def recognize_face():
    """
    Main function to perform real-time face recognition using DeepFace
    """
    global webcam_in_use, processing_active

    # Load all face embeddings
    known_embeddings = load_embeddings()
    if not known_embeddings:
        return

    if webcam_in_use:
        print("Webcam is already in use. Please wait.")
        return

    # Initialize webcam
    webcam_in_use = True
    processing_active = True
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Start recognition worker thread
    worker_thread = threading.Thread(target=recognition_worker, args=(known_embeddings,), daemon=True)
    worker_thread.start()

    frame_count = 0
    print("\nStarting face recognition...")
    print("Press 'ESC' to quit\n")

    while True:
        ret, image = webcam.read()
        if not ret or image is None:
            continue

        frame_count += 1
        display_image = image.copy()
        current_time = time.time()

        # Process every Nth frame
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            # Clear old frame from queue if it's full
            while frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except:
                    break
            try:
                frame_queue.put((copy.deepcopy(image), current_time), block=False)
            except:
                pass  # Skip frame if queue is full

        # Draw recognition results
        with recognition_lock:
            for face_key, result in recognition_results.items():
                x, y, w, h = map(int, face_key.split('_'))
                label = result['last_recognized']
                similarity = result['similarity']
                
                # Set color based on recognition status
                color = (0, 255, 0) if label != "Unknown" and label != "Error" else (0, 0, 255)
                
                # Draw rectangle
                cv2.rectangle(display_image, (x, y), (x + w, y + h), color, 2)

                # Draw face label
                text = f"{label} ({similarity:.1f}%)" if label not in ["Unknown", "Error"] else label
                cv2.putText(display_image, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Display status
        cv2.putText(display_image, "Press 'ESC' to quit", (10, display_image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Face Recognition', display_image)
        if cv2.waitKey(1) == 27:  # ESC key
            break

        # Call this periodically
        if frame_count % 1000 == 0:
            clear_memory()

    # Cleanup
    processing_active = False
    webcam.release()
    cv2.destroyAllWindows()
    webcam_in_use = False
    print("\nFace recognition stopped.")

def main():
    """
    Main entry point of the program
    """
    print("Face Recognition System")
    print("----------------------")
    print("Starting face recognition...")
    recognize_face()

def clear_memory():
    """Clear unnecessary data from memory"""
    import gc
    gc.collect()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
    

    


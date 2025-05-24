# to do : check only need 1 face, collect more data? because speed very fast with separate threading (< 1s)
# the temp2 file name change to temp_<ic>_<count>.jpg


# Import required libraries
import cv2, os
import numpy as np
from deepface import DeepFace
import json
import time
import threading
from queue import Queue
import copy
from numpy.linalg import norm

# Global configuration variables
datasets = 'datasets'  # Directory to store temporary images
embeddings_dir = 'embeddings'  # Directory to store face embeddings
imageRequired = 10  # Number of valid face embeddings to collect
webcam_in_use = False  # Flag to prevent multiple webcam access
PROCESS_EVERY_N_FRAMES = 10  # Process every 10th frame for smoother operation

# Global variables for threading
frame_queue = Queue(maxsize=1)  # Queue to pass frames between threads
detection_results = {}  # Dictionary to store detection results
detection_lock = threading.Lock()  # Lock for thread-safe access to detection_results
processing_active = True  # Flag to control worker thread
collected_embeddings = []  # List to store collected embeddings

def normalize_embedding(embedding):
    """Normalize embedding vector to unit length"""
    return embedding / norm(embedding)

def process_frame(frame, current_time):
    """
    Process a single frame for face detection and embedding generation
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

        # Clear old results at the start of new detection
        with detection_lock:
            detection_results.clear()

        # Process each detected face
        for face in face_objs:
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

                face_key = f"{x}_{y}_{w}_{h}"
                detection_results[face_key] = {
                    'bbox': (x, y, w, h),
                    'is_real': True,
                    'embedding': reps[0]['embedding'],
                    'face_img': face_img
                }

            except Exception as e:
                print(f"\n❌ [Processing Error] {str(e)}")

    except Exception as e:
        print(f"\n❌ [Detection Error] {str(e)}")

def detection_worker():
    """
    Worker thread function that processes frames from the queue
    """
    global processing_active, collected_embeddings
    print("Detection worker thread started")
    
    while processing_active:
        try:
            if not frame_queue.empty():
                frame, current_time = frame_queue.get()
                process_frame(frame, current_time)
                frame_queue.task_done()
            else:
                time.sleep(0.01)  # Small sleep when queue is empty
        except Exception as e:
            print(f"Worker thread error: {str(e)}")
            time.sleep(0.1)  # Sleep longer on error

def create_data(ic):
    """
    Main function to capture and process face images
    Args:
        ic (str): Identity Card number of the person
    Returns:
        list: Average face embedding if successful, None otherwise
    """
    global webcam_in_use, processing_active, collected_embeddings

    # Create necessary directories if they don't exist
    if not os.path.exists(datasets):
        os.makedirs(datasets)
    
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    # Check if webcam is already in use
    if webcam_in_use:
        print("Webcam is already in use. Please wait.")
        return False

    # Initialize webcam
    webcam_in_use = True
    processing_active = True
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Start detection worker thread
    worker_thread = threading.Thread(target=detection_worker, daemon=True)
    worker_thread.start()

    frame_count = 0
    count = 1

    # Main capture loop
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

        # Draw detection results and collect embeddings
        with detection_lock:
            for face_key, result in detection_results.items():
                x, y, w, h = result['bbox']
                is_real = result['is_real']
                
                # Draw rectangle and label for face
                color = (0, 255, 0) if is_real else (0, 0, 255)  # Green for real, Red for fake
                label = "Real" if is_real else "Fake"
                cv2.rectangle(display_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # If face is real and we haven't collected enough samples
                if is_real and count <= imageRequired and 'embedding' in result:
                    # Save the face image
                    temp_path2 = f"{datasets}/temp2_{count}.jpg"
                    cv2.imwrite(temp_path2, result['face_img'])
                    
                    # Add embedding to collection
                    collected_embeddings.append(result['embedding'])
                    print(f"Image {count}/{imageRequired} for IC: {ic}")
                    count += 1
            
            # Clear the detection results after processing
            detection_results.clear()

        # Display status information on screen
        cv2.putText(display_image, f"IC: {ic}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_image, f"Images collected: {count-1}/{imageRequired}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_image, "Press 'ESC' to quit", (10, display_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the processed frame
        cv2.imshow('Face Registration', display_image)
        if cv2.waitKey(1) == 27:  # ESC key to quit
            break

        if count > imageRequired:
            break

    # Cleanup
    processing_active = False
    webcam.release()
    cv2.destroyAllWindows()
    webcam_in_use = False

    # Process collected embeddings
    if collected_embeddings:
        # Calculate average embedding
        avg_embedding = np.mean(np.array(collected_embeddings), axis=0)
        avg_embedding = normalize_embedding(avg_embedding)

        # Save embedding to file
        embedding_path = os.path.join(embeddings_dir, f"{ic}_embedding.json")
        with open(embedding_path, 'w') as f:
            json.dump(avg_embedding.tolist(), f)
        print(f"✅ Embedding saved to {embedding_path}")
        return avg_embedding.tolist()
    else:
        print("❌ No valid embeddings collected.")
        return None

def main():
    """
    Main entry point of the program
    Handles user input and initiates face registration process
    """
    print("Face Registration System")
    print("----------------------")
    ic = input("Enter IC number: ")
    
    if not ic:
        print("IC number is required!")
        return
    
    result = create_data(ic)
    if result:
        print("Registration completed successfully!")
    else:
        print("Registration failed!")

if __name__ == "__main__":
    main()

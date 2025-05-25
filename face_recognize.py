"""
Face Recognition System
This script performs real-time face recognition using InsightFace and pre-trained face embeddings.

Main components:
1. Face detection using InsightFace's FaceAnalysis
2. Face alignment and embedding generation using ArcFaceONNX
3. Multi-threaded processing for real-time face recognition
4. Face embedding comparison and recognition
"""

# Import required libraries
import cv2, os  # OpenCV for image processing and os for file operations
import numpy as np  # NumPy for numerical operations
import json  # JSON for loading embeddings
import time  # Time for timestamp operations
from numpy.linalg import norm  # For vector normalization
import threading  # Threading for parallel processing
from queue import Queue  # Queue for thread-safe communication
import copy  # Copy for deep copying frames
import insightface  # Main face analysis library
from insightface.app import FaceAnalysis  # Face detection and analysis
from insightface.model_zoo.arcface_onnx import ArcFaceONNX  # Face embedding model
from insightface.utils.face_align import norm_crop  # Face alignment utility
from models.DeepFaceModel.FasNet import Fasnet
# Configuration constants - centralized settings for easy modification
CONFIG = {
    'directories': {
        'embeddings': 'embeddings',  # Directory containing face embeddings
    },
    'model': {
        'path': "models/glint360k_r50.onnx",  # Path to the face recognition model
        'detection_size': (640, 640),  # Input size for face detection
        'recognition_size': (112, 112),  # Optimized input size for glint360k model
        'detection_threshold': 0.7,  # Minimum confidence for face detection
        'face_confidence_threshold': 0.7,  # Minimum confidence for face quality
    },
    'recognition': {
        'min_similarity': 70,  # Minimum similarity percentage for recognition
        'process_every_n_frames': 1,  # Process every Nth frame for performance
    },
    'webcam': {
        'width': 640,  # Webcam capture width
        'height': 480,  # Webcam capture height
    }
}

# Global state variables - shared between threads
state = {
    'webcam_in_use': False,  # Flag to prevent multiple webcam access
    'processing_active': True,  # Flag to control worker thread
    'frame_queue': Queue(maxsize=1),  # Queue for frame processing
    'recognition_results': {},  # Dictionary to store recognition results
    'recognition_lock': threading.Lock(),  # Lock for thread-safe access to results
}

# Initialize InsightFace models
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Create face analysis app
app.prepare(ctx_id=0, det_size=CONFIG['model']['detection_size'],  # Prepare detection model
           det_thresh=CONFIG['model']['detection_threshold'])
model = ArcFaceONNX(CONFIG['model']['path'])  # Create face embedding model
model.prepare(ctx_id=0)  # Prepare embedding model
# Initialize the Fasnet model for spoofing detection
fasnet_model = Fasnet()

def normalize_embedding(embedding):
    """Normalize embedding vector to unit length"""
    return embedding / norm(embedding)  # Divide by L2 norm to get unit vector

def calculate_similarity_percentage(distance):
    """
    Convert distance to similarity percentage
    Distance of 0 = 100% similarity
    Distance of 1 = 0% similarity
    """
    return max(0, min(100, (1 - distance) * 100))  # Convert distance to percentage

def load_embeddings():
    """
    Load all face embeddings from the embeddings directory
    Returns:
        dict: Dictionary mapping IC numbers to their embeddings
    """
    embeddings = {}  # Initialize empty dictionary
    if not os.path.exists(CONFIG['directories']['embeddings']):  # Check if directory exists
        print("❌ No embeddings directory found!")
        return embeddings

    for filename in os.listdir(CONFIG['directories']['embeddings']):  # Iterate through files
        if filename.endswith('_embedding.json'):  # Check if it's an embedding file
            ic = filename.replace('_embedding.json', '')  # Extract IC number
            with open(os.path.join(CONFIG['directories']['embeddings'], filename), 'r') as f:  # Open file
                embedding = np.array(json.load(f))  # Load embedding
                embeddings[ic] = normalize_embedding(embedding)  # Normalize and store
    
    print(f"✅ Loaded {len(embeddings)} face embeddings")  # Show number of loaded embeddings
    return embeddings

def detect_spoofing(face_img, bbox=None):
    """
    Detect if the face is real or a spoof attempt using Fasnet model
    
    Args:
        face_img: Cropped face image (numpy array)
        bbox: Optional bounding box (x, y, w, h) if face_img is a full frame
    
    Returns:
        tuple: (is_real, confidence) where is_real is a boolean and confidence is a float 0-1
    """
    try:
        # If bbox is provided, analyze the face in the full frame
        if bbox is not None:
            x, y, w, h = bbox
            is_real, confidence = fasnet_model.analyze(face_img, (x, y, w, h))
        else:
            # If no bbox, assume face_img is already cropped
            is_real, confidence = fasnet_model.analyze(face_img, (0, 0, face_img.shape[1], face_img.shape[0]))
        
        # # Print detailed spoof check results
        # status = "REAL" if is_real else "FAKE"
        # print(f"\n=== Anti-Spoofing Result ===")
        # print(f"Status: {status}")
        # print(f"Confidence: {confidence*100:.1f}%")
        # print("==========================\n")
        
        return is_real, confidence
        
    except Exception as e:
        print(f"⚠️ Spoof detection error: {str(e)}")
        return False, 0.0


def process_face(face, frame, known_embeddings, face_index):
    """
    Process a single detected face
    Args:
        face: Detected face object from InsightFace
        frame: Input video frame
        known_embeddings: Dictionary of known face embeddings
        face_index: Index of the face in the current frame (1-based)
    Returns:
        dict: Recognition results for the face
    """
    x1, y1, x2, y2 = face.bbox.astype(int)  # Get face bounding box coordinates
    w, h = x2 - x1, y2 - y1  # Calculate width and height
    
    try:
        # Extract face region with padding
        padding = int(min(w, h) * 0.1)  # Calculate padding as 10% of smaller dimension
        x1_p = max(0, x1 - padding)  # Add padding to left, ensure not negative
        y1_p = max(0, y1 - padding)  # Add padding to top, ensure not negative
        x2_p = min(frame.shape[1], x2 + padding)  # Add padding to right, ensure within frame
        y2_p = min(frame.shape[0], y2 + padding)  # Add padding to bottom, ensure within frame

        # Align face using facial landmarks
        try:
            aligned_face = norm_crop(frame, landmark=face.kps)  # Align and crop face
            # Resize to optimized input size for glint360k model
            aligned_face = cv2.resize(aligned_face, CONFIG['model']['recognition_size'])

            # Add spoofing check here
            is_real, spoof_confidence = detect_spoofing(aligned_face)
            if not is_real:
                print(f"⚠️ Spoofing attempt detected! (Confidence: {spoof_confidence*100:.1f}%)")
                # Return spoof result with bounding box
                return {
                    'bbox': (x1, y1, w, h),
                    'recognized_identity': 'Spoof Detected',
                    'similarity': spoof_confidence * 100,
                    'is_spoof': True
                }

        except Exception as e:
            print(f"❌ Failed to align face: {e}")  # Handle alignment error
            return None

        # Get face embedding
        current_embedding = model.get_feat(aligned_face)  # Get face embedding
        current_embedding = normalize_embedding(current_embedding)  # Normalize embedding

        # Compare with known embeddings
        best_match = None  # Initialize best match
        best_distance = float('inf')  # Initialize best distance
        best_similarity = 0  # Initialize best similarity
        
        print(f"\nProcessing Face {face_index}:")
        print(f"Comparing with {len(known_embeddings)} stored embeddings...")

        for ic, known_embedding in known_embeddings.items():  # Compare with each known face
            distance = float(1 - np.dot(current_embedding, known_embedding))  # Calculate distance
            similarity = calculate_similarity_percentage(distance)  # Convert to percentage
            print(f"🔍 IC: {ic} - Distance: {distance:.4f}, Similarity: {similarity:.2f}%")
            
            if distance < best_distance:  # Update if better match found
                best_distance = distance
                best_match = ic
                best_similarity = similarity

        # Print recognition result
        if best_similarity >= CONFIG['recognition']['min_similarity']:
            print(f"\n✅ Face {face_index} Recognized!")
            print(f"IC: {best_match}")
            print(f"Similarity: {best_similarity:.2f}%")
            print(f"Distance: {best_distance:.4f}")
        else:
            print(f"\n❌ Face {face_index} Unknown")
            print(f"Best Match Similarity: {best_similarity:.2f}%")
            print(f"Distance: {best_distance:.4f}")
            print(f"Minimum Required Similarity: {CONFIG['recognition']['min_similarity']}%")
        print("-" * 30)

        return {
            'recognized_identity': best_match if best_similarity >= CONFIG['recognition']['min_similarity'] else "Unknown",  # Return best match if above threshold
            'bbox': (x1, y1, w, h),  # Return bounding box
            'similarity': best_similarity  # Return similarity percentage
        }

    except Exception as e:
        print(f"\n❌ [Face Processing Error] {str(e)}")  # Handle processing error
        return None

def process_frame(frame, known_embeddings, current_time):
    """
    Process a single frame for face recognition
    This function runs in a separate thread
    """
    try:
        faces = app.get(frame)  # Detect faces in frame
        print(f"\nDetected {len(faces)} face(s)")  # Show number of detected faces

        with state['recognition_lock']:  # Thread-safe access to results
            state['recognition_results'].clear()  # Clear previous results

        # Process each face in the current frame
        for i, face in enumerate(faces, 1):  # Start counting from 1
            if face.det_score < CONFIG['model']['face_confidence_threshold']:  # Check confidence
                continue

            result = process_face(face, frame, known_embeddings, i)  # Process face with current index
            if result:  # If processing successful
                face_key = f"{result['bbox'][0]}_{result['bbox'][1]}_{result['bbox'][2]}_{result['bbox'][3]}"  # Create unique key
                state['recognition_results'][face_key] = result  # Store results

    except Exception as e:
        print(f"\n❌ [Detection Error] {str(e)}")  # Handle detection error

def recognition_worker(known_embeddings):
    """
    Worker thread function that processes frames from the queue
    """
    print("Recognition worker thread started")  # Show thread start
    
    while state['processing_active']:  # Continue while processing is active
        try:
            if not state['frame_queue'].empty():  # Check if queue has frames
                frame, current_time = state['frame_queue'].get()  # Get frame from queue
                process_frame(frame, known_embeddings, current_time)  # Process frame
                state['frame_queue'].task_done()  # Mark task as complete
            else:
                time.sleep(0.01)  # Small delay if queue is empty
        except Exception as e:
            print(f"Worker thread error: {str(e)}")  # Handle worker error
            time.sleep(0.1)  # Delay on error

def draw_recognition_results(display_image, result):
    """
    Draw recognition results on the display image
    Args:
        display_image: Image to draw on
        result: Recognition result dictionary
    """
    x, y, w, h = result['bbox']  # Get bounding box
    label = result['recognized_identity']  # Get recognized label
    confidence = result.get('similarity', 0)  # Get confidence score
    is_spoof = result.get('is_spoof', False)  # Check if this is a spoof attempt
    
    # Set color and label based on status
    if is_spoof:
        color = (0, 0, 255)  # Red for spoof attempts
        text = f"Spoof Detected! ({confidence:.1f}%)"
    elif label in ["Unknown", "Error"]:
        color = (0, 0, 255)  # Red for unknown/error
        text = label
    else:
        color = (0, 255, 0)  # Green for recognized faces
        text = f"{label} ({confidence:.1f}%)"
    
    # Draw bounding box
    cv2.rectangle(display_image, (x, y), (x + w, y + h), color, 2)
    
    # Draw label with confidence
    cv2.putText(display_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def display_status(display_image):
    """
    Display status information on the image
    Args:
        display_image: Image to draw on
    """
    cv2.putText(display_image, "Press 'ESC' to quit",  # Show instructions
                (10, display_image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def clear_memory():
    """Clear unnecessary data from memory"""
    import gc  # Import garbage collector
    gc.collect()  # Run garbage collection
    cv2.destroyAllWindows()  # Close all windows

def recognize_face():
    """
    Main function to perform real-time face recognition
    """
    # Load all face embeddings
    known_embeddings = load_embeddings()  # Load known face embeddings
    if not known_embeddings:  # Check if any embeddings loaded
        return

    if state['webcam_in_use']:  # Check if webcam is in use
        print("Webcam is already in use. Please wait.")
        return

    # Initialize webcam
    state['webcam_in_use'] = True  # Set webcam in use flag
    state['processing_active'] = True  # Set processing active flag
    webcam = cv2.VideoCapture(0)  # Initialize webcam
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['webcam']['width'])  # Set width
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['webcam']['height'])  # Set height

    # Start recognition worker thread
    worker_thread = threading.Thread(target=recognition_worker,  # Create worker thread
                                   args=(known_embeddings,), daemon=True)
    worker_thread.start()  # Start worker thread

    frame_count = 0  # Initialize frame counter
    print("\nStarting face recognition...")  # Show start message
    print("Press 'ESC' to quit\n")  # Show instructions

    try:
        while True:  # Main loop
            ret, image = webcam.read()  # Read frame from webcam
            if not ret or image is None:  # Check if frame is valid
                continue

            frame_count += 1  # Increment frame counter
            display_image = image.copy()  # Create copy for display
            current_time = time.time()  # Get current time

            # Process every Nth frame
            if frame_count % CONFIG['recognition']['process_every_n_frames'] == 0:
                while state['frame_queue'].full():  # Clear queue if full
                    try:
                        state['frame_queue'].get_nowait()
                    except:
                        break
                try:
                    state['frame_queue'].put((copy.deepcopy(image), current_time), block=False)  # Add frame to queue
                except:
                    pass

            # Draw recognition results
            with state['recognition_lock']:  # Thread-safe access to results
                # Create a copy of items to avoid dictionary changed size during iteration
                results_to_draw = list(state['recognition_results'].items())
                
            # Draw results outside the lock to minimize lock time
            for face_key, result in results_to_draw:
                draw_recognition_results(display_image, result)  # Draw results

            display_status(display_image)  # Show status
            cv2.imshow('Face Recognition', display_image)  # Show frame
            
            if cv2.waitKey(1) == 27:  # Check for ESC key
                break

            # Periodically clear memory
            if frame_count % 1000 == 0:  # Every 1000 frames
                clear_memory()  # Clear memory

    finally:
        # Cleanup
        state['processing_active'] = False  # Stop processing
        webcam.release()  # Release webcam
        cv2.destroyAllWindows()  # Close windows
        state['webcam_in_use'] = False  # Reset webcam flag
        print("\nFace recognition stopped.")  # Show stop message

def main():
    """
    Main entry point of the program
    """
    print("Face Recognition System")  # Show title
    print("----------------------")  # Show separator
    print("Starting face recognition...")  # Show start message
    recognize_face()  # Start recognition

if __name__ == "__main__":
    main()  # Run main function
    
    

    


"""
Face Registration System
This script captures face images from webcam, processes them using InsightFace,
and generates face embeddings for face recognition purposes.

Main components:
1. Face detection using InsightFace's FaceAnalysis
2. Face alignment and embedding generation using ArcFaceONNX
3. Multi-threaded processing for real-time face detection
4. Face embedding collection and averaging
"""

# to do : check only need 1 face, collect more data? because speed very fast with separate threading (< 1s)
# the temp2 file name change to temp_<ic>_<count>.jpg


# Import required libraries
import cv2, os  # OpenCV for image processing and os for file operations
import numpy as np  # NumPy for numerical operations
import json  # JSON for saving embeddings
import time  # Time for timestamp operations
import threading  # Threading for parallel processing
from queue import Queue  # Queue for thread-safe communication
import copy  # Copy for deep copying frames
from numpy.linalg import norm  # For vector normalization
import insightface  # Main face analysis library
from insightface.app import FaceAnalysis  # Face detection and analysis
from insightface.model_zoo.arcface_onnx import ArcFaceONNX  # Face embedding model
from insightface.utils.face_align import norm_crop  # Face alignment utility

# Configuration constants - centralized settings for easy modification
CONFIG = {
    'directories': {
        'datasets': 'datasets',  # Directory to store temporary face images
        'embeddings': 'embeddings',  # Directory to store face embeddings
    },
    'model': {
        'path': "models/glint360k_r50.onnx",  # Path to the face recognition model
        'detection_size': (640, 640),  # Input size for face detection
        'recognition_size': (112, 112),  # Optimized input size for glint360k model
        'detection_threshold': 0.7,  # Minimum confidence for face detection
        'face_confidence_threshold': 0.7,  # Minimum confidence for face quality
    },
    'collection': {
        'required_images': 50,  # Number of face images to collect
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
    'detection_results': {},  # Dictionary to store detection results
    'detection_lock': threading.Lock(),  # Lock for thread-safe access to results
    'collected_embeddings': [],  # List to store collected face embeddings
}

# Initialize InsightFace models
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Create face analysis app
app.prepare(ctx_id=0, det_size=CONFIG['model']['detection_size'],  # Prepare detection model
           det_thresh=CONFIG['model']['detection_threshold'])
model = ArcFaceONNX(CONFIG['model']['path'])  # Create face embedding model
model.prepare(ctx_id=0)  # Prepare embedding model

def setup_directories():
    """Create necessary directories if they don't exist"""
    for directory in CONFIG['directories'].values():  # Iterate through all directories
        if not os.path.exists(directory):  # Check if directory exists
            os.makedirs(directory)  # Create directory if it doesn't exist

def normalize_embedding(embedding):
    """
    Normalize embedding vector to unit length
    Args:
        embedding: Face embedding vector
    Returns:
        Normalized embedding vector
    """
    return embedding / norm(embedding)  # Divide by L2 norm to get unit vector

def process_face(face, frame):
    """
    Process a single detected face
    Args:
        face: Detected face object from InsightFace
        frame: Input video frame
    Returns:
        dict: Processed face data including bbox, embedding, and aligned image
    """
    x1, y1, x2, y2 = face.bbox.astype(int)  # Get face bounding box coordinates
    w, h = x2 - x1, y2 - y1  # Calculate width and height
    
    # Extract face region with padding
    padding = int(min(w, h) * 0.1)  # Calculate padding as 10% of smaller dimension
    x1_p = max(0, x1 - padding)  # Add padding to left, ensure not negative
    y1_p = max(0, y1 - padding)  # Add padding to top, ensure not negative
    x2_p = min(frame.shape[1], x2 + padding)  # Add padding to right, ensure within frame
    y2_p = min(frame.shape[0], y2 + padding)  # Add padding to bottom, ensure within frame
    
    # Get landmarks and align face
    landmarks = face.kps  # Get facial landmarks
    face_img = norm_crop(frame, landmarks)  # Align and crop face using landmarks
    
    # Resize to optimized input size for glint360k model
    face_img = cv2.resize(face_img, CONFIG['model']['recognition_size'])

    # Generate embedding
    embedding = model.get_feat(face_img)  # Get face embedding
    embedding = embedding.flatten().tolist()  # Convert to 1D list
    
    return {
        'bbox': (x1, y1, w, h),  # Return bounding box
        'is_real': True,  # Flag for real face
        'embedding': embedding,  # Face embedding
        'face_img': face_img  # Aligned face image
    }

def process_frame(frame, current_time):
    """
    Process a single frame for face detection and embedding generation
    This function runs in a separate thread
    """
    try:
        faces = app.get(frame)  # Detect faces in frame
        
        with state['detection_lock']:  # Thread-safe access to results
            state['detection_results'].clear()  # Clear previous results

        for face in faces:  # Process each detected face
            if face.det_score < CONFIG['model']['face_confidence_threshold']:  # Check confidence
                continue
                
            try:
                face_data = process_face(face, frame)  # Process face
                face_key = f"{face_data['bbox'][0]}_{face_data['bbox'][1]}_{face_data['bbox'][2]}_{face_data['bbox'][3]}"  # Create unique key
                state['detection_results'][face_key] = face_data  # Store results

            except Exception as e:
                print(f"\n❌ [Processing Error] {str(e)}")  # Handle processing errors

    except Exception as e:
        print(f"\n❌ [Detection Error] {str(e)}")  # Handle detection errors

def detection_worker():
    """
    Worker thread function that processes frames from the queue.
    This runs in parallel with the main thread. It continuously checks if there are frames in the queue,
    retrieves them, and processes them for face detection/embedding.
    The queue ensures thread-safe communication between the main thread (producer) and this worker (consumer).
    """
    print("Detection worker thread started")
    
    while state['processing_active']:  # Continue while processing is active
        try:
            # Check if there is a frame available in the queue
            if not state['frame_queue'].empty():
                # Get the next frame and its timestamp from the queue (thread-safe)
                frame, current_time = state['frame_queue'].get()
                process_frame(frame, current_time)  # Process the frame
                state['frame_queue'].task_done()  # Mark task as complete
            else:
                # If no frame is available, sleep briefly to yield CPU
                time.sleep(0.01)
        except Exception as e:
            print(f"Worker thread error: {str(e)}")  # Handle worker errors
            time.sleep(0.1)  # Delay on error

def draw_detection_results(display_image, result):
    """
    Draw detection results on the display image
    Args:
        display_image: Image to draw on
        result: Detection result dictionary
    """
    x, y, w, h = result['bbox']  # Get bounding box
    is_real = result['is_real']  # Get face status
    
    color = (0, 255, 0) if is_real else (0, 0, 255)  # Green for real, red for fake
    label = "Real" if is_real else "Fake"  # Set label text
    cv2.rectangle(display_image, (x, y), (x + w, y + h), color, 2)  # Draw rectangle
    cv2.putText(display_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)  # Draw label

def display_status(display_image, ic, count):
    """
    Display status information on the image
    Args:
        display_image: Image to draw on
        ic: Identity Card number
        count: Current image count
    """
    cv2.putText(display_image, f"IC: {ic}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Show IC
    cv2.putText(display_image, f"Images collected: {count-1}/{CONFIG['collection']['required_images']}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Show progress
    cv2.putText(display_image, "Press 'ESC' to quit", 
                (10, display_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Show instructions

def save_embedding(ic):
    """
    Save collected embeddings to file
    Args:
        ic: Identity Card number
    Returns:
        list: Average embedding if successful, None otherwise
    """
    if not state['collected_embeddings']:  # Check if any embeddings collected
        print("❌ No valid embeddings collected.")
        return None

    avg_embedding = np.mean(np.array(state['collected_embeddings']), axis=0)  # Calculate average embedding
    avg_embedding = normalize_embedding(avg_embedding)  # Normalize average embedding
    avg_embedding = avg_embedding.flatten().tolist()  # Convert to list

    embedding_path = os.path.join(CONFIG['directories']['embeddings'], f"{ic}_embedding.json")  # Create file path
    with open(embedding_path, 'w') as f:  # Open file for writing
        json.dump(avg_embedding, f)  # Save embedding
    print(f"✅ Embedding saved to {embedding_path}")  # Confirm save
    return avg_embedding  # Return average embedding

def create_data(ic):
    """
    Main function to capture and process face images
    """
    if state['webcam_in_use']:  # Check if webcam is in use
        print("Webcam is already in use. Please wait.")
        return False

    setup_directories()  # Create necessary directories
    state['webcam_in_use'] = True  # Set webcam in use flag
    state['processing_active'] = True  # Set processing active flag
    
    webcam = cv2.VideoCapture(0)  # Initialize webcam
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['webcam']['width'])  # Set width
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['webcam']['height'])  # Set height

    # --- FRAME QUEUE EXPLANATION ---
    # The frame queue is a thread-safe queue (maxsize=1) shared between the main thread and detection_worker thread.
    # The main thread (here) captures frames from the webcam and places them into the queue.
    # The detection_worker thread retrieves frames from the queue and processes them for face detection/embedding.
    # This design allows the main thread to run smoothly and not be blocked by slow detection operations.
    # Only the most recent frame is kept (old frames are dropped if the queue is full), ensuring low latency.

    # Start detection worker thread
    worker_thread = threading.Thread(target=detection_worker, daemon=True)  # Create worker thread
    worker_thread.start()  # Start worker thread

    frame_count = 0  # Initialize frame counter
    count = 1  # Initialize image counter

    try:
        while True:  # Main loop
            ret, image = webcam.read()  # Read frame from webcam
            if not ret or image is None:  # Check if frame is valid
                continue

            frame_count += 1  # Increment frame counter
            display_image = image.copy()  # Create copy for display
            current_time = time.time()  # Get current time

            # --- FRAME QUEUE USAGE ---
            # Only process every Nth frame (to reduce CPU load)
            if frame_count % CONFIG['collection']['process_every_n_frames'] == 0:
                # If the queue is full, remove the oldest frame(s) to make space for the newest
                while state['frame_queue'].full():
                    try:
                        state['frame_queue'].get_nowait()  # Drop old frame
                    except:
                        break
                # Place the latest frame into the queue (non-blocking)
                try:
                    state['frame_queue'].put((copy.deepcopy(image), current_time), block=False)
                except:
                    pass

            # Process detection results
            with state['detection_lock']:  # Thread-safe access to results
                for face_key, result in state['detection_results'].items():  # Process each result
                    draw_detection_results(display_image, result)  # Draw results

                    if result['is_real'] and count <= CONFIG['collection']['required_images'] and 'embedding' in result:
                        temp_path2 = f"{CONFIG['directories']['datasets']}/temp2_{count}.jpg"  # Create temp file path
                        # cv2.imwrite(temp_path2, result['face_img'])  # Save face image
                        
                        state['collected_embeddings'].append(result['embedding'])  # Add embedding to collection
                        print(f"Image {count}/{CONFIG['collection']['required_images']} for IC: {ic}")  # Show progress
                        count += 1  # Increment counter
                
                state['detection_results'].clear()  # Clear results

            display_status(display_image, ic, count)  # Show status
            cv2.imshow('Face Registration', display_image)  # Show frame
            
            if cv2.waitKey(1) == 27 or count > CONFIG['collection']['required_images']:  # Check for exit
                break

    finally:
        # Cleanup
        state['processing_active'] = False  # Stop processing
        webcam.release()  # Release webcam
        cv2.destroyAllWindows()  # Close windows
        state['webcam_in_use'] = False  # Reset webcam flag

    return save_embedding(ic)  # Save and return embedding

def main():
    """
    Main entry point of the program
    Handles user input and initiates face registration process
    """
    print("Face Registration System")
    print("----------------------")
    ic = input("Enter IC number: ")  # Get IC number
    
    if not ic:  # Check if IC is provided
        print("IC number is required!")
        return
    
    result = create_data(ic)  # Start registration process
    if result:  # Check result
        print("Registration completed successfully!")
    else:
        print("Registration failed!")

if __name__ == "__main__":
    main()  # Run main function

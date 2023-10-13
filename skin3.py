import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from skimage import feature
import time
from deepface import DeepFace

# Streamlit settings and styles
st.set_page_config(page_title="Face Analysis", page_icon=":smiley:")

# Add an open left sidebar
st.sidebar.title("Why it matters")

# Write content in the left sidebar
st.sidebar.write("Taking care of your skin is important for both your physical and mental health. Healthy skin can help to protect you from the elements, reduce your risk of skin cancer, and boost your self-confidence. With so much information out there about skincare, it can be difficult to know where to start. Myskin.ai takes the guesswork out of skincare by providing you with personalized insights and recommendations based on your unique skin needs.")
st.sidebar.write("Your skin health journey starts today.")
st.sidebar.write("Myskin.ai is more than just a skincare app. It's your partner on your skin health journey. We're here to help you understand your skin, care for your skin, and achieve your skin goals.")
st.sidebar.write("Tell stories.")
st.sidebar.write("Share your stories of your friends. The app is free, and we do not store any images.")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .big-font {
        font-size:50px !important;
        color: #5A9;
    }
    .hover:hover {
        background-color: #f5f5f5;
        border-radius: 10px;
    }
    div[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        background-color: #f0f2f6;
        border-radius: 15px 15px 0 0;
    }
    .block-container {
        padding: 2rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Mediapipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

def compute_lbp_texture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 2
    n_points = 24
    lbp = feature.local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype('float')
    return np.sum(lbp_hist)

def draw_landmarks_with_flicker(image):
    results = face_mesh.process(image)
    landmarks_image = np.zeros_like(image, dtype=np.uint8)
    
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            for connection in mp_face_mesh.FACEMESH_TESSELATION:
                start_idx = connection[0]
                end_idx = connection[1]

                start_point = (int(landmarks.landmark[start_idx].x * image.shape[1]),
                               int(landmarks.landmark[start_idx].y * image.shape[0]))
                end_point = (int(landmarks.landmark[end_idx].x * image.shape[1]),
                             int(landmarks.landmark[end_idx].y * image.shape[0]))

                cv2.line(landmarks_image, start_point, end_point, (220, 220, 220), 1, lineType=cv2.LINE_AA)
                
                # Draw the landmark points
                cv2.circle(landmarks_image, start_point, 1, (127, 127, 127), -1)

    # Now, apply a slight blur to make the lines appear thinner
    landmarks_image = cv2.GaussianBlur(landmarks_image, (3, 3), 0)
    
    # Blend the original image with the landmarks image for a translucent effect
    alpha = 0.35
    blended_image = cv2.addWeighted(image, 1 - alpha, landmarks_image, alpha, 0)
    
    return blended_image


def count_wrinkles_and_spots(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray_roi, 9, 80, 80)
    edges = cv2.Canny(bilateral, 50, 150)
    
    wrinkles = np.sum(edges > 0)
    
    # Use adaptive thresholding
    thresh1 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Use morphological operations to fill small holes and remove small noises
    kernel = np.ones((3,3), np.uint8)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out very small contours to reduce noise
    min_spot_area = 10
    spots = len([cnt for cnt in contours if cv2.contourArea(cnt) > min_spot_area])
    
    return wrinkles, spots


def count_features(image):
    wrinkles, spots = count_wrinkles_and_spots(image)
    texture = compute_lbp_texture(image)
    return wrinkles, spots, texture

def process_image(uploaded_image):
    image = Image.open(uploaded_image).convert("RGB")
    frame = np.array(image)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    wrinkles, spots, texture = count_features(frame)
    frame = draw_landmarks_with_flicker(frame)

    return frame, wrinkles, spots, texture

# Streamlit UI
st.markdown("<div class='big-font'>Face Analysis App</div>", unsafe_allow_html=True)
st.write("Upload an image to analyze facial features.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.markdown("<div class='block-container hover'>", unsafe_allow_html=True)
    st.write("Your uploaded image:")
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True, caption="Uploaded Image.")
    
    st.write("Analyzing...")
    with st.spinner('Processing...'):
        frame, wrinkles, spots, texture = process_image(uploaded_file)
        time.sleep(10)  # Pause for 10 seconds with spinner displayed.

    st.image(frame, channels="RGB", use_column_width=True, caption="Face landmarks.")
    
    st.markdown(f"**Number of Wrinkles Detected:** {wrinkles}")
    st.markdown(f"**Number of Spots Detected:** {spots}")
    st.markdown(f"**Texture Count:** {texture}")

    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    st.write("SKIN AGE ANALYSIS.")
    # Display the uploaded image
    st.image(uploaded_file, use_column_width=True)

    # Convert the uploaded image to a format compatible with OpenCV
    image_data = uploaded_file.read()
    image_array = np.fromstring(image_data, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Initialize Mediapipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

    # Detect faces in the image
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        result = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Check if faces were detected
    if result.detections:
        for detection in result.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Extract the face region
            face_img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

            # Calculate skin age using DeepFace
            try:
                # Analyze the face image using DeepFace
                analysis = DeepFace.analyze(face_img, actions=['age'], enforce_detection=False)
                
                # Loop through each face in the result
                for face in analysis:
                    age = face['age']
                    st.write(f"Estimated skin age: {age} years")
            except Exception as e:
                st.write("Error analyzing face: " + str(e))
    else:
        st.write("No faces detected in the image.")

# Display some additional information
st.info("Note: The accuracy of age estimation may vary.")

st.write("---")
st.write("Developed with :heart: by Himika Mishra.")
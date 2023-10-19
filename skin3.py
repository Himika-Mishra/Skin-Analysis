import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from skimage import feature
import time
from deepface import DeepFace
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image
import os
import base64

# Streamlit settings and styles
st.set_page_config(page_title="Face Analysis", page_icon=":smiley:")

@st.cache
def load_image(img_path):
    return Image.open(img_path)

st.sidebar.image(load_image("myskin.png"), width=150)  # You can adjust the width as desired.

# Navbar interaction using horizontal radio buttons in the sidebar
action = st.sidebar.radio("--MENU--", ("Home", "Skin Analysis", "About Us"), key="navbar")  # Radio buttons for interaction in the sidebar

# Add an open left sidebar
st.sidebar.title("Our Motto")

# Write content in the left sidebar
st.sidebar.write("Myskin.ai is more than just a skincare app. It's your partner on your skin health journey. We're here to help you understand your skin, care for your skin, and achieve your skin goals.")
st.sidebar.write("STEP 1: Turn on your camera or upload image")
st.sidebar.write("STEP 2: Click analyse to get your report & share with friends (optional)")
st.sidebar.write("STEP 3: Discover brands that are developing products for your skin type.")

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
        display: flex;
        margin-left: auto;
        margin-right: auto;
        padding: 2rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
    }
    .big-font {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Mediapipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

# Load Model
MODEL_PATH = 'more_data(3).h5'
new_model = load_model(MODEL_PATH)

@st.cache
def compute_lbp_texture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 2
    n_points = 24
    lbp = feature.local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype('float')
    return np.sum(lbp_hist)

@st.cache
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

@st.cache
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
    min_spot_area = 4
    spots = len([cnt for cnt in contours if cv2.contourArea(cnt) > min_spot_area])
    
    return wrinkles, spots

@st.cache
def count_features(image):
    wrinkles, spots = count_wrinkles_and_spots(image)
    texture = compute_lbp_texture(image)
    return wrinkles, spots, texture

@st.cache
def process_image(uploaded_image):
    image = Image.open(uploaded_image).convert("RGB")
    frame = np.array(image)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    wrinkles, spots, texture = count_features(frame)
    frame = draw_landmarks_with_flicker(frame)

    return frame, wrinkles, spots, texture

@st.cache
def loadImage(filepath):
    test_img = tf_image.load_img(filepath, target_size=(180, 180))
    test_img = tf_image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img /= 255
    return test_img

@st.cache
def model_predict(img_path):
    global new_model
    age_pred = new_model.predict(loadImage(img_path))
    x = age_pred[0][0]
    rounded_age_value = round(x)  # Rounds 24.56 to 25
    age = 'About '+ str(rounded_age_value) +' years old'
    return age

# Streamlit UI
if action == "Home":
    # Home Page Content
    st.markdown("<div class='big-font'>MySkin.ai</div>", unsafe_allow_html=True)
    
    # Creating two divs side-by-side using HTML and CSS within Markdown
    st.markdown("""
    <div style="display: flex;">
        <div style="flex: 50%; padding: 10px; box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1); margin-right: 10px;">
            <h2>Why it matters?</h2>
            Taking care of your skin is important for both your physical and mental health. Healthy skin can help to protect you from the elements, reduce your risk of skin cancer, and boost your self-confidence. With so much information out there about skincare, it can be difficult to know where to start. Myskin.ai takes the guesswork out of skincare by providing you with personalised insights and recommendations based on your unique skin needs.
        </div>
        <div style="flex: 50%; padding: 10px; box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);">
            <h2>How it works?</h2>
            Turn on your camera or upload a picture to use our AI to analyse your skin and provide you with a report on your skin health, including:
            <ul>
                <li>Your skin type</li>
                <li>Your skin tone</li>
                <li>Your skin hydration levels</li>
                <li>Your skin elasticity</li>
                <li>The presence of any skin conditions, such as acne, rosacea, or eczema</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif action == "Skin Analysis":    
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
        
        def min_max_scale(value, min_value, max_value):
            """Scales a given value between 0 and 100 using Min-Max scaling."""
            return (value - min_value) / (max_value - min_value) * 100
    
        # Define some hypothetical maximum values for wrinkles, spots, and texture 
        # based on your dataset or domain knowledge.
        MAX_WRINKLES = 100000  # Just a placeholder value; adjust accordingly
        MAX_SPOTS = 100000     # Just a placeholder value; adjust accordingly
        MAX_TEXTURE = 100000   # Just a placeholder value; adjust accordingly
        
        # Use the min_max_scale function to scale each feature value to [0, 100]
        scaled_wrinkles = min_max_scale(wrinkles, 0, MAX_WRINKLES)
        scaled_spots = min_max_scale(spots, 0, MAX_SPOTS)
        scaled_texture = min_max_scale(texture, 0, MAX_TEXTURE)
        
        # Display the scaled scores in Streamlit
        st.markdown("<div class='block-container hover'>", unsafe_allow_html=True)
        st.markdown(f"**Standardized Wrinkles Score:** {scaled_wrinkles:.2f}")
        st.markdown(f"**Standardized Spots Score:** {scaled_spots:.2f}")
        st.markdown(f"**Standardized Texture Score:** {scaled_texture:.2f}")
    
        st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        st.write("SKIN AGE ANALYSIS.")
        st.write("Model Loaded Successfully!")
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Predicting...")
        
        # Save the uploaded file temporarily and predict
        #with open(os.path.join("tempFile.jpg"), "wb") as f:
         #    f.write(uploaded_file.getbuffer())
             
        age = model_predict(uploaded_file)
        st.write(age)
        
    # Display some additional information
    st.info("Note: The accuracy of age estimation may vary.")

elif action == "About Us":
    # Custom CSS to set font size for a specific class
    st.markdown("""
    <style>
        .quote {
            font-size:24px !important;
            font-style: italic;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)
    # About Page Content
    st.markdown("<div class='big-font'>About Us...</div>", unsafe_allow_html=True)
    # Display the quote in italics using markdown
    st.markdown('<div class="quote">"We use digital AI tool to assess your skin health and provide you with personalised insights and recommendations."</div>', unsafe_allow_html=True)

st.write("---")
st.write("Developed with :heart: by Himika Mishra.")

# Footer styles and content
footer_style = """
<style>
.footer {
  position: fixed;
  bottom: 0;
  left: 0;
  width: 100%;
  background-color: #f1f1f1;
  text-align: center;
  padding: 10px;
}
.footer-text p{
    font-size: 14px;
    font-style: italic;
}
.footer-logo {
        width: 80px; 
        height: auto;
        margin-right: 10px;
    }
</style>
"""
st.markdown(footer_style, unsafe_allow_html=True)

@st.cache
def image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
logo_base64 = image_to_base64("Hexis-Lab-Logo.png")
footer_content = f"""
<div class="footer">
    <img src="data:image/png;base64,{logo_base64}" alt="Logo" class="footer-logo">
    <span class="footer-text"><p>Validated by HexisLab Limited, The Biosphere, Draymans Way, Newcastle Helix, Newcastle, NE4 5BX | *Patent pending © 2023</p></span>
</div>
"""
st.markdown(footer_content, unsafe_allow_html=True) 

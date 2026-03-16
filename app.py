# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import rasterio
# import matplotlib.pyplot as plt

# # Load the model you downloaded from Colab
# @st.cache_resource
# def load_vssc_model():
#     return tf.keras.models.load_model("vssc_final_water_model.h5")

# model = load_vssc_model()

# st.title("🛰️ VSSC Satellite Water Detection")
# st.write("Upload a Sentinel-2 Multispectral image to see the AI analysis.")

# uploaded_file = st.file_uploader("Upload .tif image", type=["tif"])

# if uploaded_file:
#     # Processing the satellite bands
#     with rasterio.open(uploaded_file) as src:
#         # Extract B4(Red), B3(Green), B2(Blue), B8(NIR)
#         img = src.read([4, 3, 2, 8]).astype('float32') / 10000.0
        
#         # Display RGB
#         rgb = np.transpose(img[:3], (1, 2, 0))
#         rgb_viz = np.clip(rgb * 3.5, 0, 1)
        
#         # Prepare for U-Net
#         input_tensor = np.expand_dims(np.transpose(img, (1, 2, 0)), axis=0)
        
#         # Prediction
#         prediction = model.predict(input_tensor)[0].squeeze()
#         mask = (prediction > 0.5).astype(np.uint8)

#     # UI Layout
#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader("Satellite View (RGB)")
#         st.image(rgb_viz, use_container_width=True)
#     with col2:
#         st.subheader("AI Water Mask")
#         blue_mask = np.zeros((*mask.shape, 3))
#         blue_mask[mask == 1] = [0, 0.5, 1.0] # Blue tint
#         st.image(blue_mask, use_container_width=True)

#     st.info(f"Water detected in {np.mean(mask)*100:.2f}% of the area.")


import streamlit as st
import tensorflow as tf
import numpy as np
import rasterio

# --- Page Configuration ---
st.set_page_config(
    page_title=" Water Detection",
    page_icon="🛰️",
    layout="wide" 
)

# --- Custom Styling (Fixed the Typo here) ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #1E3A8A;
    }
    .stAlert {
        border-radius: 10px;
    }
    h1 {
        color: #1E3A8A;
        font-weight: 800;
    }
    </style>
    """, unsafe_allow_html=True) # Changed from unsafe_allow_stdio to unsafe_allow_html

# --- Model Loading ---
@st.cache_resource
def load_vssc_model():
    return tf.keras.models.load_model("vssc_final_water_model.h5")

try:
    model = load_vssc_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- Sidebar ---
with st.sidebar:
    # Adding a clean header to the sidebar
    st.markdown("## 🛠️ Configuration")
    st.info("Upload Sentinel-2 MS imagery for automated segmentation.")
    
    uploaded_file = st.file_uploader("Upload .tif image", type=["tif", "tiff"])
    
    st.divider()
    st.markdown("### 🎚️ Sensitivity")
    threshold = st.slider("Water Detection Threshold", 0.1, 0.9, 0.5, 0.05)
    st.caption("Lower threshold = more sensitive to water.")

# --- Main Dashboard ---
st.title("🛰️ VSSC Satellite Water Detection")
st.markdown("---")

if uploaded_file:
    with rasterio.open(uploaded_file) as src:
        num_bands = src.count
        
        if num_bands < 4:
            st.error(f"🚨 **Incompatible File:** Found only {num_bands} bands. Please upload 4-band multispectral data.")
        else:
            with st.spinner("🔄 Running U-Net Segmentation Engine..."):
                # Data Processing
                img = src.read([4, 3, 2, 8]).astype('float32') / 10000.0
                input_tensor = np.expand_dims(np.transpose(img, (1, 2, 0)), axis=0)
                
                # Inference
                prediction = model.predict(input_tensor, verbose=0)[0].squeeze()
                mask = (prediction > threshold).astype(np.uint8)
                
                # Visuals Prep
                rgb = np.transpose(img[:3], (1, 2, 0))
                rgb_viz = np.clip(rgb * 3.5, 0, 1)
                
                blue_mask = np.zeros((*mask.shape, 3))
                blue_mask[mask == 1] = [0.1, 0.4, 0.9] # Deep Blue

            # --- Analytics Row ---
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Spatial Resolution", f"{src.width}x{src.height}")
            with m2:
                water_cov = (np.mean(mask) * 100)
                st.metric("Water Coverage", f"{water_cov:.2f}%")
            with m3:
                status = "🟢 Nominal" if water_cov < 75 else "🔴 Flood Alert"
                st.metric("Region Status", status)

            # --- Comparison Row ---
            st.markdown("### 📊 Imagery Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Satellite View**")
                st.image(rgb_viz, use_container_width=True)
                
            with col2:
                st.markdown("**AI Segmented Water Bodies**")
                st.image(blue_mask, use_container_width=True)
            
            # --- Detailed System Logs ---
            with st.expander("📝 View Detailed System Logs"):
                st.code(f"""
                Input Shape: {input_tensor.shape}
                Detected Water Pixels: {np.sum(mask)}
                Mean Prediction Score: {np.mean(prediction):.4f}
                Coordinate Reference: {src.crs}
                """)
else:
    # Stylish landing state
    st.write("### ⬅️ Getting Started")
    st.warning("Please upload a .tif file from the sidebar to begin processing.")
    
    # Example placeholder images to keep layout aligned
    p1, p2 = st.columns(2)
    p1.image("https://via.placeholder.com/600x400/eeeeee/999999?text=Input+Imagery", use_container_width=True)
    p2.image("https://via.placeholder.com/600x400/eeeeee/999999?text=Segmentation+Mask", use_container_width=True)
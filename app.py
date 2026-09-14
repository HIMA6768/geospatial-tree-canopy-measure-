import streamlit as st
import folium
from folium.plugins import Draw, Geocoder
from streamlit_folium import st_folium
import shapely.geometry
import numpy as np
import cv2
import io
import base64
import mercantile
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
import xml.etree.ElementTree as ET
import shapely.geometry

# ---  OPENCV PIPELINE ---
def canopy_detect(img,mask):
    val=0
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if 0.5<area<2000:
         cv2.drawContours(img, [cnt], -1, (255, 0,255), 1)
         val+=1
    return val,img


def detect_trees(img):
    _, mask_img= cv2.threshold(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph_img=cv2.morphologyEx(mask_img,cv2.MORPH_OPEN,kernel=kernel,iterations=1)
    masked_img=cv2.bitwise_and(img,img,mask=morph_img)
    treecount,final_img=canopy_detect(img.copy(),morph_img)
    return treecount,final_img,masked_img

# --- INPUT PROCESSOR FUNCTIONS ---
def process_map_selection(poly_coords):
    """Function for handling Map Polygon Selection & Tile Fetching"""
    try:
        polygon = shapely.geometry.Polygon(poly_coords)
        minx, miny, maxx, maxy = polygon.bounds  
        st.info("Fetching exact tiles for selected map area...")
        
        zoom = 17
        tiles = list(mercantile.tiles(minx, miny, maxx, maxy, zoom))
        if not tiles:
            st.error("No tiles found for this boundary!")
            return None
            
        min_x = min([t.x for t in tiles])
        max_x = max([t.x for t in tiles])
        min_y = min([t.y for t in tiles])
        max_y = max([t.y for t in tiles])
        
        width_tiles = (max_x - min_x + 1)
        height_tiles = (max_y - min_y + 1)
        tile_size = 256
        
        mosaic = np.zeros((height_tiles * tile_size, width_tiles * tile_size, 3), dtype=np.uint8)
        for t in tiles:
            url = f"https://mt1.google.com/vt/lyrs=y&x={t.x}&y={t.y}&z={zoom}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                t_arr = np.frombuffer(response.content, np.uint8)
                t_img = cv2.imdecode(t_arr, cv2.IMREAD_COLOR)
                if t_img is not None:
                    px = (t.x - min_x) * tile_size
                    py = (t.y - min_y) * tile_size
                    mosaic[py:py+tile_size, px:px+tile_size] = t_img
                
        img_np = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
        b_min = mercantile.bounds(min_x, max_y, zoom)
        b_max = mercantile.bounds(max_x, min_y, zoom)
        
        height, width, bands = img_np.shape
        transform = rasterio.transform.from_bounds(b_min.west, b_min.south, b_max.east, b_max.north, width, height)
        
        with MemoryFile() as memfile:
            with memfile.open(driver='GTiff', height=height, width=width, count=bands, dtype=img_np.dtype, crs='EPSG:4326', transform=transform) as dataset:
                dataset.write(np.moveaxis(img_np, -1, 0))
                geom = [shapely.geometry.mapping(polygon)]
                out_image, _ = mask(dataset, geom, crop=True)
                cropped_np = np.moveaxis(out_image, 0, -1)
                
        # Convert RGB back to BGR for OpenCV pipeline compatibility
        return cv2.cvtColor(cropped_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Error processing map area: {e}")
        return None

def process_uploaded_image(uploaded_file):
    """Function for handling direct image file uploads"""
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name.lower()
    
    # Check if the uploaded file is a TIFF image
    if file_name.endswith(('.tif', '.tiff')):
        try:
            with MemoryFile(file_bytes) as memfile:
                with memfile.open() as dataset:
                    # Read image bands (Rasterio reads as CHW: Channels, Height, Width)
                    img_array = dataset.read()
                    
                    if img_array.shape[0] >= 3:
                        # Take first 3 bands (RGB) and move axis to HWC (Height, Width, Channels)
                        img_rgb = np.moveaxis(img_array[:3], 0, -1)
                    else:
                        # If grayscale or single band, convert to 3-channel
                        img_rgb = np.stack([img_array[0]] * 3, axis=-1)
                        
                    # Normalize if 16-bit TIFF to 8-bit for OpenCV processing
                    if img_rgb.dtype != np.uint8:
                        img_rgb = cv2.normalize(img_rgb, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        
                    # Convert RGB to BGR since OpenCV expects BGR format
                    opencv_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    return opencv_image
        except Exception as e:
            st.error(f"Error reading TIFF file: {e}")
            return None
    else:
        # Standard formats (PNG, JPG, JPEG, WEBP) via OpenCV imdecode
        file_bytes_np = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes_np, cv2.IMREAD_COLOR)
        return opencv_image

def process_uploaded_kml(kml_file):
    """Function placeholder for handling KML file boundary parsing"""
    try:
        file_bytes = kml_file.read()
        root = ET.fromstring(file_bytes)
        
        coords_list = []
        
        # XML namespace handle korar jonno tag-er sesh অংশ (local name) check kora
        for elem in root.iter():
            tag_name = elem.tag.split('}')[-1] # Namespace prefix bad dewa
            if tag_name == 'coordinates':
                text = elem.text
                if text:
                    points = []
                    for line in text.strip().split():
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            try:
                                lon, lat = float(parts[0]), float(parts[1])
                                points.append((lon, lat))
                            except ValueError:
                                continue
                                
                    if len(points) >= 3:  # Valid polygon require kore minimum 3 points
                        coords_list.append(points)
                        
        if not coords_list:
            st.error("No valid polygon coordinates found in the KML file!")
            return None
            
        # Prothom polygon-er coordinates-gulo return kora
        return coords_list[0]
        
    except Exception as e:
        st.error(f"Error parsing KML file: {e}")
        return None


# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Tree Counting Pipeline", layout="wide")
st.title("🌲 Automated  Tree Counting & Canopy Analysis")

# Initialize Session States
if 'center_coord' not in st.session_state:
    st.session_state.center_coord = [22.5726, 88.3639]
if 'poly_coords' not in st.session_state:
    st.session_state.poly_coords = None
if 'raw_image' not in st.session_state:
    st.session_state.raw_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'masked_image' not in st.session_state:
    st.session_state.masked_image = None
if 'tree_count' not in st.session_state:
    st.session_state.tree_count = None

# --- SECTION 1: 3 Block Selection Layout in a Row ---
st.markdown("### Choose Input Mode")
input_mode = st.radio(
    "Select Input Source",
    options=["Select from Map", "Upload Image", "Upload KML File"],
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)


if 'last_input_mode' not in st.session_state:
    st.session_state.last_input_mode = input_mode

if st.session_state.last_input_mode != input_mode:
    st.session_state.poly_coords = None
    st.session_state.raw_image = None
    st.session_state.processed_image = None
    st.session_state.masked_image = None
    st.session_state.tree_count = None
    st.session_state.last_input_mode = input_mode
    st.rerun()

st.markdown("---")


    # ... baki map-er code ...

# --- SECTION 2: Dynamic UI Rendering Based on Mode ---
if input_mode == "Select from Map":
    st.subheader("🗺️ Draw Region of Interest on Google Hybrid Map")
    m_input = folium.Map(location=st.session_state.center_coord, zoom_start=15)
    
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Hybrid',
        overlay=False,
        control=True
    ).add_to(m_input)
    
    Geocoder(position='topright', collapsed=False, placeholder='Search location...').add_to(m_input)
    
    draw = Draw(
        export=False,
        position='topleft',
        draw_options={
            'polyline': False,
            'polygon': True,
            'rectangle': True,
            'circle': False,
            'marker': False,
            'circlemarker': False
        }
    )
    m_input.add_child(draw)
    folium.LayerControl().add_to(m_input)
    
    map_data = st_folium(m_input, width=1600, height=600, key="input_map")
    
    if map_data and map_data.get("last_active_drawing"):
        drawing = map_data["last_active_drawing"]
        geom_type = drawing["geometry"]["type"]
        coords = drawing["geometry"]["coordinates"]
        st.session_state.poly_coords = coords[0] if geom_type in ["Polygon", "Rectangle"] else coords
        st.success(f"Boundary Selected Successfully via {geom_type}!")

    if st.session_state.poly_coords is not None:
        if st.button("🚀 Process Selected Map Area"):
            with st.spinner("Fetching tiles and running pipeline..."):
                img_bgr = process_map_selection(st.session_state.poly_coords)
                if img_bgr is not None:
                    st.session_state.raw_image = img_bgr
                    count, final_img, masked_img = detect_trees(img_bgr)
                    st.session_state.tree_count = count
                    st.session_state.processed_image = final_img
                    st.session_state.masked_image = masked_img

elif input_mode == "Upload Image":
    st.subheader("📤 Upload Aerial Image File")
    uploaded_file = st.file_uploader("Choose an image file (PNG, JPG, WEBP, TIFF)", type=["png", "jpg", "jpeg", "webp", "tif", "tiff"])
    
    st.markdown("Or quick-test with pre-loaded samples:")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    # Session state initialization for sample selection
    if 'active_sample' not in st.session_state:
        st.session_state.active_sample = None

    with col1:
        if st.button("Use Sample Image 1"):
            st.session_state.active_sample = "assets/image1.png"
            st.rerun()
    img_bgr = None
    
    # Handle file upload vs sample selection
    if uploaded_file is not None:
        st.session_state.active_sample = None  # Clear sample if user uploads a file
        img_bgr = process_uploaded_image(uploaded_file)
    elif st.session_state.active_sample is not None:
        try:
            img_bgr = cv2.imread(st.session_state.active_sample)
            if img_bgr is None:
                st.error(f"Could not load sample image from {st.session_state.active_sample}")
        except Exception as e:
            st.error(f"Error loading sample: {e}")
            
    # Process and display if image is available
    if img_bgr is not None:
        st.session_state.raw_image = img_bgr
        
        # Display preview of image
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Raw Image Preview", width=500)
        
        if st.button("🚀 Process Image"):
            with st.spinner("Running canopy detection pipeline..."):
                count, final_img, masked_img = detect_trees(img_bgr)
                st.session_state.tree_count = count
                st.session_state.processed_image = final_img
                st.session_state.masked_image = masked_img
                st.rerun()
elif input_mode == "Upload KML File":
    st.subheader("📂 Upload KML Boundary File")
    kml_file = st.file_uploader("Choose a KML file", type=["kml"])
    
    st.markdown("Or quick-test with pre-loaded samples:")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    # Session state initialization for sample KML
    if 'active_sample_kml' not in st.session_state:
        st.session_state.active_sample_kml = None

    with col1:
        if st.button("Use Sample KML 1"):
            st.session_state.active_sample_kml = "assets/image1.kml"
            st.rerun()
            
    poly_coords = None
    
    # Handle KML file upload vs sample KML selection
    if kml_file is not None:
        st.session_state.active_sample_kml = None  # Clear sample if custom file is uploaded
        poly_coords = process_uploaded_kml(kml_file)
    elif st.session_state.active_sample_kml is not None:
        try:
            with open(st.session_state.active_sample_kml, "rb") as f:
                poly_coords = process_uploaded_kml(f)
        except Exception as e:
            st.error(f"Error loading sample KML: {e}")
            
    if poly_coords is not None:
        st.session_state.poly_coords = poly_coords
        st.success("Polygon coordinates extracted from KML!")
        
        if st.button("🚀 Process KML Boundary"):
            with st.spinner("Fetching tiles from KML boundary and running pipeline..."):
                img_bgr = process_map_selection(st.session_state.poly_coords)
                
                if img_bgr is not None:
                    st.session_state.raw_image = img_bgr
                    # Area limits gulo pass kora holo jate kono argument mismatch na hoy
                    count, final_img, masked_img = detect_trees(img_bgr)
                    st.session_state.tree_count = count
                    st.session_state.processed_image = final_img
                    st.session_state.masked_image = masked_img
                    st.rerun()
# --- SECTION 3: Results Display Panel ---
if st.session_state.processed_image is not None:
    st.markdown("---")
    st.subheader("📊 Pipeline Execution Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB), caption=" Canopy Area estimation", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(st.session_state.masked_image, cv2.COLOR_BGR2RGB), caption="Canopy Mask ", use_container_width=True)
        
    st.metric(label="Estimated Tree Count", value=f"{st.session_state.tree_count} Trees")
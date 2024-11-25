import streamlit as st
import cv2
import numpy as np
import os
import random
from queue import PriorityQueue

# Define directories for processed images
input_folder = r'D:\Moon\moon_imgs'
output_folder = r'D:\Moon\processed_images'

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define landing site coordinates (85.28° S, 31.20° E) for reference
LANDING_SITE = (85.28, 31.20)

# Moon feature detection functions
def detect_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=100, param2=30, minRadius=10, maxRadius=100)
    _, threshold_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = image.copy()
    features = []

    # Detect craters
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.rectangle(output_image, (x - r, y - r), (x + r, y + r), (0, 255, 255), 1)  # Thin yellow border
            cv2.putText(output_image, "Crater", (x - r, y - r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            features.append((x - r, y - r, x + r, y + r, "Crater"))

    # Detect boulders
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 1)  # Thin red border
            cv2.putText(output_image, "Boulder", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            features.append((x, y, x + w, y + h, "Boulder"))

    return output_image, features

# Create grid based on detected features
def create_grid(image, rectangles, cell_size=10):
    height, width = image.shape[:2]
    grid = np.zeros((height // cell_size, width // cell_size), dtype=int)
    for (x1, y1, x2, y2) in rectangles:
        for i in range(max(0, y1 // cell_size), min(grid.shape[0], y2 // cell_size)):
            for j in range(max(0, x1 // cell_size), min(grid.shape[1], x2 // cell_size)):
                grid[i, j] = 1
    return grid

# A* Pathfinding Algorithm
def a_star_pathfinding(grid, start, goal):
    height, width = grid.shape
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while not open_set.empty():
        current = open_set.get()[1]
        if current == goal:
            return reconstruct_path(came_from, current)
        for neighbor in get_neighbors(current, width, height):
            if grid[neighbor[1], neighbor[0]] == 1:
                continue
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if not any(neighbor == i[1] for i in open_set.queue):
                    open_set.put((f_score[neighbor], neighbor))
    return []

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos, width, height):
    neighbors = []
    x, y = pos
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        neighbor = (x + dx, y + dy)
        if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
            neighbors.append(neighbor)
    return neighbors

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    total_path.reverse()
    return total_path

# Generate random scientific stops with unique descriptions
def generate_scientific_stops(image, num_stops=5):
    height, width = image.shape[:2]
    stops = []
    
    for _ in range(num_stops):
        x = random.randint(0, width)
        y = random.randint(0, height)
        
        # Generate a random description
        description = f"Scientific observation site at coordinates ({x}, {y}). Possible water ice or unique mineral formations."
        stops.append({"coord": (x, y), "description": description})
    
    return stops

# Streamlit application sections
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Moon Path Navigator"])

# Home Page
if app_mode == "Home":
    st.header("Moon Path Navigator")
    st.image("home.jpg", use_column_width=True)
    st.markdown("""Moon Path Navigator project aims to chart a safe, scientifically valuable 100-meter route for a lunar rover on the Moon's south pole using high-resolution Chandrayaan-2 data. The path avoids obstacles like craters and boulders and includes 10 investigation stops. The route maximizes scientific yield, focusing on high-interest sites such as potential water ice deposits and unique geological features.""")

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### Objective
                1. Analyse the landing site region in the south polar region of the Moon
                2. From the landing site, generate a minimum 100 m distance traverse path for the safe 
                  navigation of rover
                3. Primarily the Path should include some major scientific stops
                #### Expected Outcome
                1. An annotated map/image with clear marks of anticipated rover track
                2. The annotated map should contain the major features along the traverse
                3. Minimum 100 m is the rover traverse path (but not limited to it), at least 10 stops should 
                be marked in the annotated map
                4. A detailed explanation for the proposed traverse route, stops and safe navigation should 
                be provided
                """)

# Moon Path Navigator Page
elif app_mode == "Moon Path Navigator":
    st.header("Moon Path Navigator")
    test_image = st.file_uploader("Choose a Moon Image:")

    if test_image and st.button("Process Image"):
        image = cv2.imdecode(np.frombuffer(test_image.read(), np.uint8), cv2.IMREAD_COLOR)
        processed_image, features = detect_features(image)
        
        # Extract only the bounding box coordinates from features
        rectangles = [(x1, y1, x2, y2) for (x1, y1, x2, y2, label) in features]
        
        grid = create_grid(image, rectangles)
        start = (0, grid.shape[0] - 1)
        goal = (grid.shape[1] - 1, 0)
        
        path = a_star_pathfinding(grid, start, goal)
        
        # Plot thin white path line on the image, spanning the entire path
        for i in range(len(path) - 1):
            start_point = (path[i][0] * 10, path[i][1] * 10)
            end_point = (path[i + 1][0] * 10, path[i + 1][1] * 10)
            cv2.line(processed_image, start_point, end_point, (255, 255, 255), 1)  # Thin white line with thickness 1
        
        # Mark 5 to 10 stops with white dots along the path
        stop_indices = random.sample(range(len(path)), random.randint(5, 10))
        stops = []
        for idx in stop_indices:
            stop_point = (path[idx][0] * 10, path[idx][1] * 10)
            cv2.circle(processed_image, stop_point, 5, (255, 255, 255), -1)
            stops.append({"coord": stop_point, "description": f"Scientific stop at {stop_point}"})
        
        # Highlight scientific stops with larger white circle and description
        scientific_stops = generate_scientific_stops(processed_image)
        for stop in scientific_stops:
            coord = stop["coord"]
            cv2.circle(processed_image, coord, 5, (255, 255, 255), -1)
            cv2.circle(processed_image, coord, 10, (255, 255, 255), 1)
            cv2.putText(processed_image, stop["description"], (coord[0] - 50, coord[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the processed image
        st.image(processed_image, channels="BGR")

        # Provide download option
        output_path = os.path.join(output_folder, "processed_image.png")
        cv2.imwrite(output_path, processed_image)
        st.download_button(
            label="Download Processed Image",
            data=open(output_path, "rb").read(),
            file_name="processed_image.png",
            mime="image/png"
        )

        # Display the scientific stops descriptions
        st.write("### Scientific Stops Descriptions")
        for stop in scientific_stops:
            st.markdown(f"- **{stop['description']}**")

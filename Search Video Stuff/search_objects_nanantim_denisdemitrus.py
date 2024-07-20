import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import streamlit as st

# Loading the Pre-trained Inception V3 Model
model = tf.keras.applications.InceptionV3(weights='imagenet')

# Preprocessing the function for Inception V3
preprocess_input = tf.keras.applications.inception_v3.preprocess_input

# Decoding the predictions
decode_predictions = tf.keras.applications.inception_v3.decode_predictions

# Sets the maximum size for videos
# which is based on the size in the
# config.toml file
# If the size is changed here, it should
# be changed in the config.toml file as well
max_file_size_mb = 10
max_size_bytes = max_file_size_mb * 1024 * 1024

# Displays the title of the site
st.title("Video Object Detection Using Inception V3")

# Uploading the video
video = st.file_uploader("Please upload your video", 
                         type=["mp4", "mov", "avi", "mkv"])

# Input search query
query = st.text_input("Search for an object in the video")


# Makes sure the video does not exceed the file size
def validate_size(uploaded_file):

  # Checks the size of the file to be sure it
  # doesn't exceed the stated size
  if uploaded_file.size > max_size_bytes:

    # Error message to show the size is larger than the accepted size
    st.error(f"File size exceeeds {max_file_size_mb} MB. Please uplaod a smaller file.")
    return False
  
  # The file is within the accepted boudaries
  return True


# Function to process the video
def process_video(video_path, search_query):
  cap = cv2.VideoCapture(video_path)
  frame_count = 0
  found_frames = []

  while cap.isOpened():

    # Reads a frame from the video
    ret, frame = cap.read()

    # Breaks the loop if no frame is read (end of video)
    if not ret:
      break

    # Increments the frame count
    frame_count += 1

    # Convert frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the frame to the size Inception V3 accepts
    resized_frame = cv2.resize(rgb_frame, (299, 299))

    # Converts the frame to a numpy array
    img_array = np.array(resized_frame)
    
    # Adds a batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocesses the frame for the model
    img_array = preprocess_input(img_array)

    # Time to predict the objects in the frame
    predictions = model.predict(img_array)

    # Decodes the top ten predictions
    decoded_predictions = decode_predictions(predictions, top=10)[0]

    # Loops through the predictions
    for prediction in decoded_predictions:

      # Check if the search query is found in the frame
      if search_query.lower() in prediction[1].lower():

        # If it's found, it stores the frame adn it's number
        found_frames.append((frame_count, rgb_frame))
        break

  # Releases the video capture object
  cap.release()

  # Showing the results we have
  if found_frames:

    # Shows the success message for the frames that were found
    st.success(f"Found '{search_query}' in the following frames:")

    # Loops through the found frames
    for frame_num, frame in found_frames:

      # Displays the frame
      st.image(frame, caption = f"Frame {frame_num}", use_column_width=True)
  
  else:

    # Tells the user there is no such object
    st.error("Object doesn't exit!!!")

# Checks if the Search button is clicked
if st.button("Search"):

  # Checks if the video is uploaded and the search query is entered
  if video is not None and query:

    # Checks if the file size is valid
    if validate_size(video):

      # Save the uploaded video
      with open("uploaded_video.mp4", "wb") as f:
        f.write(video.getbuffer())

      # Process the video
      process_video("uploaded_video.mp4", query)

  else:

    # Shows that there is no file uploaded or no search query
    st.error("Please uplaod a video and enter a search query.")
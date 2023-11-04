from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import cv2
import mediapipe as mp
import numpy as np
import io
from RNN_model import *
from collections import deque
import torch

# Initialize Flask app
app = Flask(__name__)

# Configure CORS to allow requests from the specified origin
cors = CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5000"}})

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Create a buffer to hold the last 40 frames of skeleton data
frame_buffer = deque(maxlen=40)


# Function to extract skeleton information from a frame
def get_skeleton_info(frame):
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)

    # If landmarks are found, extract and return them
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        frame_data = []
        for landmark in landmarks:
            frame_data.extend([landmark.x, landmark.y, landmark.z])
        return frame_data
    # If no landmarks are found, return a zero-filled list
    else:
        return [0] * 33 * 3


# Define the predict route to handle prediction requests
@app.route('/predict', methods=['POST'])
@cross_origin(origin='http://127.0.0.1:5000')
def predict():
    # Check if the POST request has a file part
    if request.files:
        # Read the file from the request
        file = request.files['frame']
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)

         # Convert the file to a numpy array
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)

        # Decode the image data to a frame
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # Extract skeleton data from the frame
        skeleton_data = get_skeleton_info(frame)

        # Add the skeleton data to the frame buffer
        frame_buffer.append(skeleton_data)

        # If the buffer is full, run the prediction
        if len(frame_buffer) == 40:
            model_input = torch.tensor(list(frame_buffer))
            print(model_input)

            # If the input is all zeros, clear the buffer and return 0
            if torch.all(model_input.eq(0)):
                frame_buffer.clear()
                return jsonify({'result': 0})
            else:

                # Initialize the model
                model = RNN()

                # Load the pre-trained model weights
                model.load_state_dict(torch.load('rnn_epoch73_loss0.19.pth', map_location=torch.device('cpu')))

                # Set the model to evaluation mode
                model.eval()

                # Make a prediction
                with torch.no_grad():
                    output = model(model_input)
                result = output.max(0, keepdim=True)
                print(result)

                # Get the index of the max result
                max_indices = result.indices.item()

                # Clear the buffer for the next set of frames
                frame_buffer.clear()
                print(type(max_indices))
                return jsonify({'result': max_indices})
        else:
            return jsonify({'result': 'Frame received'})
    else:
        return jsonify({'error': 'No file received'})

# Run the Flask app on the specified host and port
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=False)

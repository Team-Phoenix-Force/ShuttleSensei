# ShuttleSensei

## Badminton AI Coach - README

This project builds an AI coach for badminton singles, analyzing gameplay videos to provide insights and coaching recommendations.

Features
Point and Winner Prediction: Predicts the outcome of rallies using a combination of 2D/3D CNNs and an enhanced transformer model.
Shot Detection: Identifies the type of shot played by each player in every rally.
Player Analysis: Analyzes player strengths, weaknesses, and potential strategies based on the AI's insights.
Technologies
Backend:
Python
PyTorch (Deep Learning framework)
OpenCV (Computer Vision library)
Flask (Web framework)
Frontend:
React (JavaScript framework)
Dataset
The AI model is trained on several hours of badminton video data annotated by human experts. This data includes:

Video footage of badminton singles matches
Annotations for:
Point outcomes (win/loss)
Shot types by each player (e.g., clear, smash, drop)
Player positions on the court
Shuttlecock location
Usage
Upload a Badminton Match Video: The user uploads a video of a badminton singles match (duration: 40 minutes to 2 hours) through the web interface.
Backend Processing:
The backend receives the video and initiates processing using Flask.
Processing involves:
Video pre-processing with OpenCV
Shot detection and player tracking using the AI model
Generating player analysis based on the AI's results
Email Notification: Once processing is complete, the user receives an email notification.
Results Visualization:
The user can access the web interface to view the AI-generated insights, including:
Predicted points and winner
Shot detection by player for each rally
Player strengths, weaknesses, and potential strategies
Dependencies
This project requires the following dependencies to be installed:

Python (>=3.7)
PyTorch with appropriate GPU support (if applicable)
OpenCV
Flask
React
Refer to the respective documentation for installation instructions.

Getting Started
Clone the Repository:

Bash
git clone https://github.com/Team-Phoenix-Force/ShuttleSensei.git 
Use code with caution.
Install Dependencies:

Bash
cd Badminton-AI-Coach
pip install -r requirements.txt
Use code with caution.
Run the Application:

Bash
python backend/app.py
Use code with caution.
(This starts the backend Flask server)

Frontend Setup:
Follow the instructions in the frontend directory to set up the React application.

# ShuttleSensei - Badminton AI Coach

This project aims to develop an AI coach for badminton singles matches, offering analysis and coaching recommendations by processing gameplay videos.

## Features

- **Point and Winner Prediction:** Utilizes a combination of 2D/3D CNNs and an enhanced transformer model to predict rally outcomes.
- **Shot Detection:** Identifies shot types played by each player during rallies.
- **Player Analysis:** Analyzes player strengths, weaknesses, and suggests strategies based on AI insights.

## Technologies

**Backend:**
- Python
- PyTorch (Deep Learning framework)
- OpenCV (Computer Vision library)
- Flask (Web framework)

**Frontend:**
- React (JavaScript framework)

## Dataset

The AI model is trained on annotated badminton video data, including:
- Video footage of badminton singles matches
- Annotations for point outcomes and shot types by each player
- Player positions on the court
- Shuttlecock location

## Usage

1. **Upload a Badminton Match Video:** Users upload a video of a badminton singles match (duration: 40 minutes to 2 hours) via the web interface.
2. **Backend Processing:** The backend receives the video, processes it using Flask, performs:
    - Video pre-processing with OpenCV
    - Shot detection and player tracking using the AI model
    - Generating player analysis
3. **Email Notification:** Upon completion of processing, users receive an email notification.
4. **Results Visualization:** Users access the web interface to view AI-generated insights, including:
    - Predicted points and winner
    - Shot detection by player for each rally
    - Player strengths, weaknesses, and potential strategies



Refer to respective documentation for installation instructions.

## Getting Started

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/Team-Phoenix-Force/ShuttleSensei.git
    ```

2. **Install Dependencies:**
    ```bash
    cd Badminton-AI-Coach
    pip install -r requirements.txt
    ```

3. **Run the Application:**
    ```bash
    python backend/app.py
    ```

    - This starts the backend Flask server.

4. **Frontend Setup:**
    Follow instructions in the frontend directory to set up the React application.

**Note:** Use code with caution.

## Research Papers

A collection of research papers related to badminton AI coaching is available in the `references` folder of the repository. Some notable papers include:

- "[Actionformer](https://arxiv.org/abs/2202.07925)" - Actionformer: Transformer-Based Framework for Action Recognition
- "[SlowFastNetwork](https://arxiv.org/abs/1812.03982)" - SlowFast Networks for Video Recognition
- "[VisionTransformer](https://paperswithcode.com/method/vision-transformer)" - Vision Transformer
- "[Attention is all you Need](https://arxiv.org/abs/1706.03762)" - Attention is All You Need

Please refer to the respective papers for more detailed information.






# Real-time Object Counting with YOLOv8 and OpenCV
Overview
This Python application utilizes the YOLOv8 object detection model along with OpenCV to perform real-time object counting. The application is capable of detecting objects in a video stream and counting the number of objects present within a predefined detection area.

Features
Real-time object detection using YOLOv8.
Object counting within a specified detection area.
Adjustable webcam resolution.
Easy-to-use command-line interface.
Installation
Clone the repository:

git clone https://github.com/your_username/your_repo.git
Install the required dependencies:

pip install -r requirements.txt
Usage
Navigate to the project directory:

cd your_repo
Run the application with the desired webcam resolution:

python main.py --resolution 1280 720
Position the object of interest within the camera frame, preferably in the left corner.
The application will start counting the detected objects within the specified detection area.
Press the Esc key to exit the application.

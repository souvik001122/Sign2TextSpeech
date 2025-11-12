# Sign2TextSpeech: Real-Time ASL Recognition and Speech

## Overview

Sign2TextSpeech is a real-time system that translates American Sign Language (ASL) finger-spelling into text and speech using computer vision and deep learning. The project leverages a convolutional neural network (CNN) and MediaPipe hand tracking to robustly recognize hand gestures from a webcam feed, even in challenging backgrounds and lighting.

---

## Key Features

- **Live ASL Recognition:** Detects and classifies hand gestures from webcam video in real time.
- **Robust to Environment:** Uses MediaPipe landmarks and skeletonization to minimize background and lighting issues.
- **Text and Speech Output:** Recognized gestures are displayed as text and can be spoken aloud using text-to-speech.
- **User-Friendly:** Designed for both the Deaf/Hard-of-Hearing community and those unfamiliar with sign language.

---

## How It Works

1. **Hand Detection & Landmark Extraction:**  
   The webcam captures hand images. MediaPipe extracts 21 hand landmarks, which are drawn on a blank image to create a “skeleton” representation.

2. **Preprocessing:**  
   The skeleton image is resized and normalized for input to the CNN, making the system robust to background clutter and lighting.

3. **Gesture Classification:**  
   A custom-trained CNN predicts the ASL letter or group from the skeleton image. To improve accuracy, similar-looking letters are grouped and then further distinguished using landmark math.

4. **Text & Speech Output:**  
   The recognized character is appended to a sentence. Users can click a button to hear the sentence spoken aloud.

---

## Technical Details

- **Model:**  
  - CNN trained on 180+ skeleton images per alphabet letter.
  - 26 letters grouped into 8 classes for higher accuracy.
  - Achieves up to 97% accuracy in varied conditions.

- **Libraries:**  
  - Python 3.12+
  - OpenCV, MediaPipe, NumPy, Keras, TensorFlow, pyttsx3

- **Hardware:**  
  - Standard webcam

---

## System Flow

1. **Data Acquisition:**  
   Vision-based (webcam) hand capture, no gloves or special hardware needed.

2. **Preprocessing:**  
   - Hand detection (MediaPipe)
   - Landmark extraction
   - Skeleton drawing (OpenCV)
   - Image normalization

3. **Classification:**  
   - CNN predicts gesture group
   - Landmark math for final letter

4. **Output:**  
   - Display as text
   - Optional speech (pyttsx3)

---

## Getting Started

1. Clone the repo and install requirements.
2. Download the model file (see instructions in the repo).
3. Run the app:
   ```
   python final_pred.py
   ```
   or for the web version:
   ```
   streamlit run app.py
   ```
4. Allow webcam access and start signing!

---

## Why This Project?

Sign2TextSpeech bridges the communication gap for Deaf/Hard-of-Hearing individuals and those unfamiliar with ASL. It’s a practical, real-time tool for inclusive communication, education, and accessibility.

---

## 👨‍💻 Author
Souvik Das  
GitHub: [github.com/souvik001122](https://github.com/souvik001122)  
Email: 231210104@nitdelhi.ac.in  
B.Tech CSE, NIT Delhi



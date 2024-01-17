import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image

# Load the trained model
model = load_model('currency_detection_model.h5')
class_indices = {0: 'Fake', 1: 'Real'}  # Define class indices

def predict_currency(img):
    test_image = cv2.resize(img, (255, 255))
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    prediction = class_indices[int(result[0][0])]
    return prediction

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        prediction = predict_currency(cv2.imread(file_path))
        result_label.config(text=f"PREDICTION: {prediction}")

def capture_image():
    cap = cv2.VideoCapture(0)
    
    while True:
        _, frame = cap.read()
        cv2.imshow("Captured Image", frame)

        if cv2.waitKey(1) == 13:
            break
    
    cap.release()
    cv2.destroyAllWindows()

    # Preprocess the captured frame
    test_image = cv2.resize(frame, (255, 255))
    test_image = np.expand_dims(test_image, axis=0)

    # Print the raw prediction result
    result = model.predict(test_image)
    print(f"Raw prediction result: {result}")

    # Convert the result to the predicted class
    prediction = class_indices[int(result[0][0])]
    print(f"Converted prediction result: {prediction}")
    result_label.config(text=f"Prediction: {prediction}")

def close_window():
    root.destroy()  # Close the Tkinter window

# GUI
root = tk.Tk()
root.title("Currency Detection")
root.attributes('-fullscreen', True)  # Maximize the window to full size

# Set the background color to beige
root.config(bg="#f5f5dc")

# header widget
header_label = tk.Label(root, text="DETECTION OF INDIAN COUNTERFEIT CURRENCY USING CONVOLUTION NEURAL NETWORK", font=("Arial", 23, "bold"), bg="#f5f5dc")
header_label.place(relx=0.5, rely=0.1, anchor="center")

# Paragraph
paragraph_text = "WELCOME"
paragraph_label = tk.Label(root, text=paragraph_text, font=("Arial", 15), bg="#f5f5dc")
paragraph_label.place(relx=0.5, rely=0.25, anchor="center")

browse_button = tk.Button(root, text="Browse", command=browse_image, width=20, height=2, font=("Arial", 14, "bold"))
browse_button.place(relx=0.5, rely=0.35, anchor="center")

# Load a camera icon image
camera_icon_image = Image.open("camera.png")
camera_icon_image = camera_icon_image.resize((50, 50))  # Resize the image as needed
camera_icon_image = ImageTk.PhotoImage(camera_icon_image)

# Create a button with the camera icon
capture_button = tk.Button(root, image=camera_icon_image, command=capture_image, width=50, height=50)
capture_button.place(relx=0.5, rely=0.45, anchor="center")

result_label = tk.Label(root, text="PREDICTION: ", font=("Arial", 18, "bold"), bg="#f5f5dc")
result_label.place(relx=0.5, rely=0.6, anchor="center")

exit_button = tk.Button(root, text="Exit", command=close_window, width=10, font=("Arial", 12, "bold"))
exit_button.place(relx=0.95, rely=0.95, anchor="se")  # Position the button at the bottom right

root.mainloop()

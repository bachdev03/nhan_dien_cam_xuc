import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


import numpy as np
from keras.src.saving import load_model
from keras.src.utils import load_img, img_to_array

# Cấu hình
MODEL_PATH = 'gan_generator.h5'
TARGET_SIZE = (48, 48)  # Kích thước ảnh phù hợp với mô hình
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Hàm dự đoán cảm xúc
def predict_emotion(image_path):
    model = load_model(MODEL_PATH)
    image = load_img(image_path, target_size=TARGET_SIZE, color_mode="grayscale")
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_emotion = EMOTIONS[np.argmax(prediction)]
    return predicted_emotion

# Xử lý khi chọn ảnh
def select_image():
    file_path = filedialog.askopenfilename(
        title="Chọn một ảnh",
        filetypes=[("Image files", "*.jpg;*.png;*.jpeg")]
    )
    if file_path:
        # Hiển thị ảnh lên giao diện
        image = Image.open(file_path)
        image = image.resize((200, 200))  # Chỉnh kích thước hiển thị
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

        # Dự đoán cảm xúc
        emotion = predict_emotion(file_path)
        result_label.config(text=f"Cảm xúc dự đoán: {emotion}")

# Tạo giao diện
root = tk.Tk()
root.title("Dự đoán cảm xúc từ ảnh")
root.geometry("400x400")

# Nút chọn ảnh
select_button = tk.Button(root, text="Chọn ảnh", command=select_image, font=("Arial", 14))
select_button.pack(pady=20)

# Vùng hiển thị ảnh
image_label = tk.Label(root)
image_label.pack(pady=10)

# Nhãn hiển thị cảm xúc
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()

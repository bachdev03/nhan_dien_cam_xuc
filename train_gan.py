import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

# Đảm bảo sử dụng đúng phiên bản TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Khai báo latent_dim
latent_dim = 100  # Chiều dài của vector ngẫu nhiên

# 1. Định nghĩa đường dẫn
TRAIN_DIR = 'D:/Pyhton/Bai_tap_nhom_6/fer2013/train'
TEST_DIR = 'D:/Pyhton/Bai_tap_nhom_6/fer2013/test'

if not os.path.exists(TRAIN_DIR):
    print(f"Thư mục huấn luyện không tồn tại: {TRAIN_DIR}")
else:
    print(f"Đường dẫn hợp lệ: {TRAIN_DIR}")

# 2. Tiền xử lý dữ liệu
def load_data(train_dir, test_dir, target_size=(48, 48), batch_size=64):
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        color_mode="grayscale",  # Chuyển ảnh thành grayscale
        batch_size=batch_size,
        class_mode="categorical"
    )
    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        color_mode="grayscale",  # Chuyển ảnh thành grayscale
        batch_size=batch_size,
        class_mode="categorical"
    )
    return train_data, test_data

# 3. Xây dựng Generator
def build_generator(latent_dim):
    model = Sequential([
        Input(shape=(latent_dim,)),  # Lớp Input không nên có input_dim
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dense(128 * 12 * 12),
        BatchNormalization(),
        LeakyReLU(0.2),
        Reshape((12, 12, 128)),
        Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2DTranspose(1, kernel_size=4, strides=1, padding='same', activation='tanh')
    ])
    return model

# 4. Xây dựng Discriminator (Thêm lớp Flatten)
def build_discriminator():
    model = Sequential([
        Input(shape=(48, 48, 1)),
        Conv2D(64, kernel_size=4, strides=2, padding='same'),
        LeakyReLU(0.2),
        Dropout(0.4),
        Conv2D(128, kernel_size=4, strides=2, padding='same'),
        LeakyReLU(0.2),
        Dropout(0.4),
        Flatten(),  # Thêm lớp Flatten sau các lớp Conv2D
        Dense(1, activation='sigmoid')
    ])
    return model

# 5. Xây dựng GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))  # Sử dụng latent_dim cho GAN input
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    return Model(gan_input, gan_output)

# 6. Huấn luyện GAN
def train_gan(generator, discriminator, gan, train_data, epochs=1000, batch_size=64):
    latent_dim = 100
    half_batch = batch_size // 2
    train_data = iter(train_data)  # Chuyển train_data thành iterator

    for epoch in range(epochs):
        # Lấy dữ liệu thật
        real_images, _ = next(train_data)

        real_images = real_images[:half_batch]  # Lấy một nửa batch
        real_images = real_images.astype(np.float32)
        real_labels = np.ones((half_batch, 1), dtype='float32')


        # Sinh dữ liệu giả
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_images = generator.predict(noise)
        fake_images = fake_images.astype(np.float32)  # Chuyển fake_images thành float32
        fake_labels = np.zeros((half_batch, 1), dtype='float32')

        # Kiểm tra kích thước của real_images và fake_images
        assert real_images.shape[1:] == (48, 48, 1), f"Kích thước real_images không hợp lệ: {real_images.shape}"
        assert fake_images.shape[1:] == (48, 48, 1), f"Kích thước fake_images không hợp lệ: {fake_images.shape}"

        # Huấn luyện Discriminator
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Huấn luyện Generator thông qua GAN
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1), dtype='float32')
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Hiển thị tiến trình
        if epoch % 100 == 0:
            print(f"{epoch}/{epochs} [D loss: {d_loss}] [G loss: {g_loss}]")

# 7. Main execution
if __name__ == "__main__":
    # Load dữ liệu từ thư mục
    train_data, test_data = load_data(TRAIN_DIR, TEST_DIR)

    # Xây dựng mô hình GAN
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()

    # Biên dịch mô hình discriminator
    discriminator.compile(optimizer=Adam(0.0002), loss='binary_crossentropy', metrics=['accuracy'])

    # Xây dựng và biên dịch GAN
    gan = build_gan(generator, discriminator)
    gan.compile(optimizer=Adam(0.0002), loss='binary_crossentropy')

    # Huấn luyện mô hình
    train_gan(generator, discriminator, gan, train_data, epochs=1000, batch_size=64)

    # Lưu mô hình
    BASE_DIR = 'D:/Pyhton/Bai_tap_nhom_6'  # Đảm bảo BASE_DIR đúng
    generator.save(os.path.join(BASE_DIR, "gan_generator.h5"))
    print("Generator đã được lưu thành công!")

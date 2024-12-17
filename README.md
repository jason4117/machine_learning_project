# **Convolutional Neural Network (CNN) for Image Classification**

## **Project Overview**
This project implements a **Convolutional Neural Network (CNN)** to classify images of **cats** and **dogs**. The model uses the **Keras** library with TensorFlow as its backend. The dataset includes two sets of images:
- **Training set**: 1000 images of cats and 1000 images of dogs.
- **Test set**: Images for evaluation to measure model performance.

---

## **Table of Contents**
1. [Features](#features)  
2. [Dataset Structure](#dataset-structure)  
3. [Technologies Used](#technologies-used)  
4. [Setup and Installation](#setup-and-installation)  
5. [Project Structure](#project-structure)  
6. [How to Run the Project](#how-to-run-the-project)  
7. [Results](#results)  
8. [Customization](#customization)  

---

## **Features**
- Image preprocessing (rescaling, augmentation).
- CNN architecture with Convolutional, Pooling, Flattening, and Dense layers.
- Training and validation of the model using a split dataset.
- Evaluation of model accuracy and loss on test data.
- Single image prediction capability.

---

## **Dataset Structure**
The dataset should be organized as follows:

```
dataset/
│
├── training_set/
│   ├── cats/       # 1000 images of cats
│   ├── dogs/       # 1000 images of dogs
│
└── test_set/
    ├── cats/       # Test images of cats
    ├── dogs/       # Test images of dogs
```

Ensure images are labeled and stored correctly under respective folders for training and testing.

---

## **Technologies Used**
- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

---

## **Setup and Installation**

1. **Clone the Repository**  
   ```bash
   git clone <repository-link>
   cd <project-folder>
   ```

2. **Install Required Libraries**  
   Install the necessary Python packages using `pip`:
   ```bash
   pip install tensorflow keras numpy matplotlib pillow
   ```

3. **Prepare the Dataset**  
   - Place the dataset into the `dataset/` directory.
   - Follow the structure mentioned in the [Dataset Structure](#dataset-structure) section.

4. **Run the Script**  
   You can execute the project using a Jupyter Notebook or Python script.

---

## **Project Structure**
The project follows this directory layout:

```
CNN_Image_Classification/
│
├── dataset/
│   ├── training_set/     # Training images
│   └── test_set/         # Testing images
│
├── cnn_image_classifier.ipynb  # Main Jupyter Notebook with code
├── single_prediction/          # Directory for single prediction images
│
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

---

## **How to Run the Project**

### Step 1: **Data Preprocessing**
Preprocess the images:
- Rescale pixel values to `[0, 1]`.
- Apply augmentation (zoom, flip, shear) for training data.

```python
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
```

### Step 2: **Build the CNN**
Construct the CNN model using Keras:
```python
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```

### Step 3: **Compile and Train**
Compile and train the model:
```python
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(training_set, validation_data=test_set, epochs=25)
```

### Step 4: **Make Predictions**
Predict a single image:
```python
from tensorflow.keras.preprocessing import image
import numpy as np

test_image = image.load_img('single_prediction/cat_or_dog.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)

if result[0][0] == 1:
    print("It's a Dog!")
else:
    print("It's a Cat!")
```

---

## **Results**
- **Training Accuracy**: ~90% (Example)
- **Validation Accuracy**: ~85%
- **Loss**: Reduced after sufficient epochs.

You can evaluate the model's performance by observing the training and validation accuracy.

Check out this medium post for more information!
https://medium.com/@szhang15/building-an-image-classifier-with-cnns-a-step-by-step-guide-a4de2d90946f

---

## **Customization**
To optimize or modify the model:
1. **Change the Architecture**:  
   - Increase the number of filters or layers for better accuracy.
   - Modify the dense layer units.

2. **Tune Hyperparameters**:  
   - Change `epochs`, `batch_size`, or `learning_rate`:
     ```python
     cnn.fit(training_set, validation_data=test_set, epochs=50, batch_size=64)
     ```

3. **Adjust Augmentation**:  
   Modify augmentation parameters:
   ```python
   shear_range=0.1, zoom_range=0.1, horizontal_flip=False
   ```

4. **Input Image Size**:  
   Increase the input resolution for complex images:
   ```python
   target_size=(128, 128)
   ```

---

## **License**
This project is open-source and free to use.

---

## **Contact**
For questions or suggestions, feel free to contact me:

- **Email**: [szhang15@laurentian.ca]  
- **GitHub**: [jason4117]  

---


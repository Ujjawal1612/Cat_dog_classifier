
README — Cats vs Dogs Classifier
Problem Statement
The project aims to build a Convolutional Neural Network (CNN) to classify images as either cats or dogs. This helps understand the core concepts of CNNs, TensorFlow, and Keras for image classification tasks.

Dataset Details
The dataset is sourced from Kaggle (https://www.kaggle.com/datasets/tongpython/cat-and-dog). It contains two main folders:

training_set: Images of cats and dogs organized in separate subfolders, used for training and validation with an 80%-20% split.

test_set: Separate folder with unseen cat and dog images used for final testing.

All images are resized to 128x128 pixels during preprocessing.

Approach
Images are loaded via Keras’ image_dataset_from_directory function.

The dataset is split into training (80%) and validation (20%) subsets.

Pixel values are normalized to the range 0–1.

The CNN model consists of two convolutional layers each followed by max pooling layers, a flatten layer, a dense layer with ReLU activation, and an output sigmoid layer for binary classification.

The model is compiled with binary crossentropy loss, Adam optimizer, and accuracy metric.

Trained for 15 epochs.

Training and validation accuracy and loss curves are plotted for performance visualization.

The model is evaluated on the separate test set.

Predictions on random images from validation and test sets are visualized with actual and predicted labels.

Results
The CNN achieves approximately 70–80% accuracy on validation and test sets, meeting project expectations.

Training and validation curves indicate effective learning without severe overfitting.

Challenges
Correctly organizing the dataset with proper folder structure for Keras’s image loader.

Preventing overfitting with a straightforward CNN architecture.

Ensuring consistent image resizing and normalization.

Managing computational resources during training.

Learnings
Practical knowledge of CNNs applied to real-world image classification.

Hands-on experience with TensorFlow and Keras APIs for dataset handling, model building, training, and evaluation.

Importance of dataset preprocessing and splitting.

Visualization of model performance and predictions.

Understanding binary classification workflow end-to-end.

Resources
TensorFlow and Keras documentation

Kaggle Cats vs Dogs dataset

Matplotlib for plotting

Coursera Deep Learning specialization, Andrew Ng

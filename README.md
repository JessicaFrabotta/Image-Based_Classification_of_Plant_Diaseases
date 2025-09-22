🌿 Image-Based Plant Disease Classification
This project focuses on the classification of plant diseases from images, a crucial task in agriculture to prevent the spread of pandemics and improve crop health. The model is a multi-class classifier capable of identifying 30 different disease classes.

💻 Dataset
The project uses a combination of two datasets for training:

PlantVillage Dataset: I selected an augmented version of the popular PlantVillage dataset, which includes 60,342 images and 38 classes.

Custom Dataset: I created my own dataset of 15,157 images with 16 classes, personally taking the photos and getting help from my parents, who are farmers, for data labeling.

The initial goal was to combine the entire dataset, but due to memory issues, I had to select the classes with the largest number of instances, resulting in a final dataset composed of 7 classes from my dataset and the rest from PlantVillage. All images were augmented and have a size of 256x256 pixels.

🛠️ Model
For the classification, I implemented a neural network based on TensorFlow and Keras, with distributed training via Horovod. The model is a very simple Convolutional Neural Network (CNN) with about 1.6 million parameters. I also developed a custom loss function, the Weighted Categorical Cross-Entropy, to handle classes with an unbalanced number of instances.

📈 Results
As expected, the Convolutional Neural Network (CNN) showed the best performance. An interesting result was also from the Random Forest with Featurization and Transfer Learning on small images (32x32), showing unexpected effectiveness despite the sensitivity of this type of task to image size.

⏭️ Future Work
There are several areas for future experimentation and improvement:

Use larger networks like AlexNet with larger images to improve performance.

Combine the two original datasets to get a much larger dataset (over 75,000 images and 54 classes).

In general, having more computational resources would allow for further exploration and optimization.

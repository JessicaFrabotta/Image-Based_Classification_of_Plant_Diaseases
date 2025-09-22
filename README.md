# 🌿 Image-Based Plant Disease Classification

This project focuses on the **classification of plant diseases from images**, a crucial task in agriculture to prevent the spread of pandemics and improve crop health.  
The model is a **multi-class classifier** capable of identifying **30 different disease classes**.

---

## 💻 Dataset

The training data comes from the combination of two datasets:

- **PlantVillage Dataset**  
  - Augmented version of the popular dataset  
  - **60,342 images**  
  - **38 classes**

- **Custom Dataset**  
  - Personally created with **15,157 images** and **16 classes**  
  - Photos taken by myself and labeled with the help of my parents (farmers)  
  - Due to memory constraints, only the **7 largest classes** were included in the final dataset  

🔹 Final dataset: mixed from both sources, with all images **augmented** and resized to **256×256 pixels**.

---

## 🛠️ Model

- Implemented with **TensorFlow** and **Keras**  
- Distributed training using **Horovod**  
- Architecture: **Convolutional Neural Network (CNN)**  
  - ~ **1.6 million parameters**  
- Custom **Weighted Categorical Cross-Entropy loss** to handle unbalanced classes  

---

## 📈 Results

- The **CNN** achieved the best performance as expected  
- Surprisingly, a **Random Forest with Featurization + Transfer Learning** on small images (32×32) also showed notable effectiveness, despite the sensitivity of this task to image size  

---

## ⏭️ Future Work

Areas for further improvement and experimentation:

- Use **larger networks** (e.g., AlexNet) with higher resolution images  
- Combine the **two full datasets** → ~75,000 images and 54 classes  
- Leverage **greater computational resources** for deeper optimization and experimentation  

---

## 📌 Project Highlights

- ✅ 30 plant disease classes  
- ✅ Custom dataset built with real-world farming expertise  
- ✅ CNN with custom loss for class imbalance  
- ✅ Experiments with both DL and traditional ML methods  

---

## 📜 License

This project is released under the **MIT License**.

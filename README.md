# Word Classification Using SVM through MFCC

📄 Project Overview
This repository demonstrates word classification from speech data using Mel-Frequency Cepstral Coefficients (MFCC) and Support Vector Machine (SVM). The primary goal is to extract features from speech audio files, train an SVM classifier, and evaluate its performance for recognizing words.

🛠️ Key Features

- **MIT License**: This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
- **Librosa**: Python package for audio analysis and feature extraction.
- **scikit-learn**: For SVM classification and model evaluation.
- **Matplotlib & Seaborn**: Libraries for visualizing data and results.

---

## 📥 Setup Instructions

### 1. Mount Google Drive

To access the dataset on Google Drive, use the following:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install Required Libraries
Install the necessary libraries:
```python
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
```

---

## 📂 Dataset Structure

The dataset contains **MFCC** files organized by class (word). The directory structure is:
```
/content/drive/MyDrive/project_17_18/MFCC_Dataset
```

---

## 📊 Data Processing

### Loading MFCC Files
The `.MFC` files are loaded and parsed for each word class:
```python
data = []
labels = []

for class_folder in os.listdir(dataset_dir):
    class_folder_path = os.path.join(dataset_dir, class_folder)
    for file_name in os.listdir(class_folder_path):
        if file_name.endswith('.MFC'):
            with open(file_path, 'r') as file:
                data_values = [float(line.strip()) for line in file.readlines()[1:]]
                data.append(data_values)
                labels.append(class_folder)
```

### Visualizing MFCC Data
To visualize how each word's MFCC features look:
![Image](https://github.com/user-attachments/assets/6d5e85ab-9b82-4ea8-9775-5fa2e2e1a9cd)
---

## 🧠 Model Training & Evaluation

### Splitting Data
Split the dataset into training and testing sets:
```python
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
```

### SVM Model Training
Train the **SVM classifier** with a linear kernel:
```python
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train, y_train)
```

### Performance Evaluation
Evaluate the model’s **accuracy**:
```python
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

#### Confusion Matrix
Visualize the classification performance using a confusion matrix:
![Image](https://github.com/user-attachments/assets/4b3c5147-0708-46e7-9178-634e1c490778)

---

## 💾 Saving the Model
Save the trained model for future use:
```python
import joblib
joblib.dump(svm_classifier, '/content/drive/MyDrive/project_17_18/svm_model.pkl')
```

---

## 📈 Evaluation Metrics
Performance metrics include:
- **Accuracy**: The overall correct predictions.
- **Precision**: The proportion of true positive predictions.
- **Recall**: How well the model captures positive instances.
- **Loss**: Calculated as `1 - accuracy`.

### Example Output:
```plaintext
Accuracy: 95.67%
Precision: 94.50%
Recall: 96.30%
Loss: 4.33%
```

---

## 🔮 Future Work
- **Advanced Models**: Experiment with other models such as **RNNs**, **LSTMs**, or **CNNs** for enhanced performance.
- **Data Augmentation**: Implement noise addition, pitch shifting, and more to diversify the dataset.
- **Expanded Dataset**: Add more words and increase the dataset for better generalization.

---

## 📜 References
- **Librosa**: Python package for audio analysis and feature extraction.
- **scikit-learn**: For SVM classification and model evaluation.
- **Matplotlib & Seaborn**: Libraries for visualizing data and results.

---

## 📝 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
```

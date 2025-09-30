# 🌸 Iris Dataset K-Nearest Neighbors (KNN) Classifier

This project demonstrates how to use the Iris dataset to train and evaluate a **K-Nearest Neighbors (KNN)** classifier using Python's `scikit-learn`. It also includes feature normalization, performance evaluation, and visualization of decision boundaries.

---

## 📦 Dependencies

Make sure the following Python packages are installed:

pip install kagglehub scikit-learn pandas matplotlib numpy
📁 Dataset
We use the Iris dataset, downloaded via kagglehub.

Download Code:
python
Copy code
import kagglehub

# Download dataset
path = kagglehub.dataset_download("uciml/iris")
📊 Features
The Iris dataset contains 150 samples with the following features:

Sepal Length

Sepal Width

Petal Length

Petal Width

Target Classes:
Iris-setosa

Iris-versicolor

Iris-virginica

🧠 Project Steps
Download and load dataset using kagglehub.

Preprocess the dataset:

Extract numeric features

Encode labels as integers

Normalize features using StandardScaler

Split into training and testing sets using train_test_split

Train a KNeighborsClassifier with different values of K

Evaluate using:

Accuracy score

Confusion matrix (visualized)

Visualize decision boundaries using the first two features

📈 Results
You will see:

Accuracy scores for different K values (K = 1, 3, 5, 7, 9)

Confusion matrices for each model

Decision boundary plot (2D using 2 features)

📉 Decision Boundary
A 2D decision boundary is plotted using only:

Sepal length

Sepal width

For better accuracy across all features, consider using PCA for 2D reduction.

📂 Files
├── iris_knn_classifier.py    # Main Python script
├── README.md                 # Project documentation
📚 References
scikit-learn documentation

Kaggle: Iris Dataset

kagglehub PyPI

✅ To Do
 Add PCA visualization for 4D decision boundary

 Deploy as a Streamlit or Flask web app

 Add CLI options for custom K and dataset path

👨‍💻 Author
Made with ❤️ using Python and scikit-learn.

# K-Nearest Neighbors Classifier with Hyperparameter Tuning

## Overview
This project implements a K-Nearest Neighbors (KNN) classifier to predict heart disease using a dataset. It explores different distance metrics and data normalization techniques to improve classification performance. The project also includes hyperparameter tuning to select the best value of `k` and the most effective distance function.

## What I Learned
### Distance Metrics in KNN
One of the key aspects of the KNN algorithm is how distances between data points are measured. In this project, I implemented and compared the following distance functions (found in `utils.py`):

1. **Euclidean Distance** - Measures straight-line distance between two points in space. It is widely used in real-world applications like image recognition and clustering.
2. **Minkowski Distance (p=3)** - A generalization of Euclidean distance that allows control over the importance of different dimensions.
3. **Cosine Similarity Distance** - Measures the cosine of the angle between two vectors, commonly used in text analysis and recommendation systems.
<img width="698" alt="Screenshot 2025-02-25 at 2 33 00 PM" src="https://github.com/user-attachments/assets/276ecf2e-39d9-4eb3-aec5-e09042459372" />

By experimenting with these distance functions, I gained insight into how different metrics affect model performance depending on the dataset characteristics.

### Feature Scaling Techniques
Feature scaling is crucial in distance-based algorithms to prevent features with larger magnitudes from dominating calculations. I implemented two types of scaling in `utils.py`:

1. **Min-Max Scaling** - Normalizes feature values to the range [0,1].
2. **Normalization Scaling** - Converts each feature vector into a unit vector.

These techniques significantly improved classification performance by ensuring that all features contribute equally to the distance computations.

### Hyperparameter Tuning and Prediction
To optimize the KNN model, I performed hyperparameter tuning in `utils.py` using the `HyperparameterTuner` class. This involved testing various values of `k` (ranging from 1 to 29) and different distance functions. The best model was selected based on F1-score, ensuring a balance between precision and recall. The tuning process considered:
- The impact of distance functions on classification accuracy.
- The influence of feature scaling on performance.
- Choosing the best `k` value to avoid underfitting and overfitting.

### F1 Score and Prediction
The F1-score, implemented in `utils.py`, was used to evaluate model performance by balancing precision and recall. Prediction in KNN was implemented in `knn.py`, where:
- **get_k_neighbors()** retrieves the nearest neighbors of a data point.
- **predict()** assigns a class label based on the majority vote of the nearest neighbors.

These functions are crucial for making accurate predictions and assessing the effectiveness of different hyperparameters.

## Implementation Details
- **Data Processing**: Implemented in `data.py`, where the dataset is preprocessed and split into training, validation, and test sets.
- **KNN Model**: Defined in `knn.py`, including training, nearest neighbor search, and prediction methods.
- **F1 Score Calculation**: Implemented in `utils.py` to evaluate classification performance.
- **Testing and Execution**: Conducted in `test.py`, where different configurations are evaluated.

## Results and Findings
After running experiments, the optimal KNN model was determined based on the highest F1-score. The findings include:
- **Without Scaling:** Euclidean distance performed best with an optimal `k` value.
- **With Scaling:** Min-Max Scaling further improved performance, highlighting the importance of feature normalization in KNN.
- The choice of `k` and distance function significantly influenced results, reinforcing the importance of hyperparameter tuning.

## Conclusion
This project deepened my understanding of how different distance metrics affect classification performance and the importance of feature scaling in distance-based models. Additionally, tuning hyperparameters like `k` and distance functions is crucial for achieving the best results. 




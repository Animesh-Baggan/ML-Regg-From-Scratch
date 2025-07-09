# ML-Regg-From-Scratch
ML Regeression From scratch python Code


# Linear Regression from Scratch

This project demonstrates the implementation of Linear Regression from scratch using NumPy, comparing it with scikit-learn's built-in LinearRegression implementation.

## 📋 Overview

The project showcases:
- Loading and preprocessing the diabetes dataset
- Implementing Linear Regression using the Normal Equation method
- Comparing custom implementation with scikit-learn's LinearRegression
- Model evaluation using R² score

## 🚀 Features

### Custom Linear Regression Class (`MeraLR`)
- **Normal Equation Implementation**: Uses the mathematical formula β = (X^T X)^(-1) X^T y
- **Manual Coefficient Calculation**: Computes intercept and coefficients without using optimization algorithms
- **scikit-learn Compatible Interface**: Implements `fit()` and `predict()` methods similar to scikit-learn

### Key Components
1. **Data Loading**: Uses sklearn's diabetes dataset
2. **Train-Test Split**: 80-20 split with random state for reproducibility
3. **Model Comparison**: Custom implementation vs scikit-learn's LinearRegression
4. **Performance Evaluation**: R² score comparison

## 📊 Dataset

- **Dataset**: Diabetes dataset from scikit-learn
- **Features**: 10 features (age, sex, BMI, blood pressure, etc.)
- **Target**: Disease progression measure
- **Samples**: 442 patients
- **Shape**: X (442, 10), y (442,)

## 🔧 Implementation Details

### Mathematical Foundation
The Linear Regression model is implemented using the **Normal Equation**:
```
β = (X^T X)^(-1) X^T y
```

Where:
- `X` is the feature matrix (with bias term added)
- `y` is the target vector
- `β` contains the coefficients (including intercept)

### Custom Class Structure

```python
class MeraLR:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X_train, y_train):
        # Add bias term (intercept)
        X_train = np.insert(X_train, 0, 1, axis=1)
        
        # Calculate coefficients using Normal Equation
        betas = np.linalg.inv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(y_train)
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]
    
    def predict(self, X_test):
        y_pred = np.dot(X_test, self.coef_) + self.intercept_
        return y_pred
```

## 📈 Results

### Model Performance
- **Custom Implementation R² Score**: 0.4399
- **scikit-learn R² Score**: 0.4399
- **Perfect Match**: Both implementations produce identical results

### Model Parameters
- **Intercept**: 151.8833
- **Coefficients**: 10 feature coefficients ranging from -895.55 to 861.13

## 🛠️ Usage

### Prerequisites
```bash
pip install numpy scikit-learn
```

### Running the Code
1. Load the Jupyter notebook: `ML Regg From Scratch.ipynb`
2. Execute cells sequentially
3. Compare results between custom and scikit-learn implementations

### Key Code Sections

```python
# Load data
X, y = load_diabetes(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Custom implementation
lr = MeraLR()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Evaluate
r2_score(y_test, y_pred)
```

## 🎯 Learning Objectives

This project demonstrates:
1. **Mathematical Understanding**: Implementation of Normal Equation
2. **NumPy Operations**: Matrix operations, inversions, and dot products
3. **Object-Oriented Programming**: Creating custom ML classes
4. **Model Validation**: Comparing custom implementations with established libraries
5. **Linear Algebra**: Understanding matrix operations in ML

## 🔍 Key Insights

- **Normal Equation**: Provides exact solution (no iterations needed)
- **Computational Complexity**: O(n³) for matrix inversion, suitable for small datasets
- **Numerical Stability**: May face issues with singular or near-singular matrices
- **Scalability**: Not suitable for very large datasets due to memory constraints

## 📚 Mathematical Background

### Linear Regression Formula
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

### Normal Equation Derivation
The Normal Equation is derived by setting the derivative of the cost function to zero:
```
∇J(β) = 0
```

This leads to:
```
β = (X^T X)^(-1) X^T y
```

## 🤝 Contributing

Feel free to contribute by:
- Adding gradient descent implementation
- Implementing regularization (Ridge/Lasso)
- Adding cross-validation
- Improving documentation

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: This implementation is for educational purposes. For production use, consider using established libraries like scikit-learn, which include optimizations and additional features.

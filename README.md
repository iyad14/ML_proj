# Extending the Machine Learning Framework

This guide explains how to add a new machine learning model to the existing framework by inheriting from the `BaseModel` class. This ensures that all models maintain a consistent interface and functionality.

## Prerequisites

Before you start, make sure you are familiar with:
- Python programming
- Basic machine learning concepts
- The scikit-learn library

## Step 1: Create Your Model File

1. Create a new Python file in the `models/` directory.
2. Name the file according to your model, e.g., `your_model.py`.

## Step 2: Import Required Modules

At the top of your file, import the necessary modules. You must import the `BaseModel` class from the base model file:

```python
from .base_model import BaseModel
from sklearn.something import YourModelClassifier  # Import your specific model class from scikit-learn or another library
```

## Step 3: Define Your Model Class

Create a new class that inherits from `BaseModel`:

```python
class YourModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
```

## Step 4: Implement Required Methods

Implement the `fit` and `test_configurations` methods. The `fit` method should train the model on the training data, and `test_configurations` should find the best model configuration:

```python
def fit(self, param1=default_value):
    self.model = YourModelClassifier(param1=param1)
    self.model.fit(self.X_train, self.y_train)

def test_configurations(self):
    # Implement your logic to test different configurations and store the best result in self.results
```

## Step 5: (Optional) Override Other Methods

If needed, you can override other methods from the `BaseModel` class, such as `evaluate`, `save_model`, or `load_model`.

## Step 6: Testing Your Model

To test your new model, import and use it in the `main.py` script, similar to how other models are used:

```python
from models.your_model import YourModel

# In the appropriate section of main.py
model = YourModel(X_train, X_test, y_train, y_test)
best_config = model.run()
print(best_config)
```

## Step 7: Document Your Model

Provide documentation for your model class, methods, and any important logic to help others understand and effectively use your model.

---
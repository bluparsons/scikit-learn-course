<p align="center">
  <img 
    src="https://github.com/user-attachments/assets/483d6899-06fe-428b-a1c3-5273432a77f5"
    alt="banner"
    width="50%"
  />
</p>

This course provides descriptions and examples of all scikit-learn tools, the course is divided into the steps that they would be used in a machine learning project: 

1. Installation
2. Load data
3. Pre-processing
4. Feature engineering
5. Dimensionality reduction
6. Machine learning: model selection
7. Machine learning: regression
8. Machine learning: classification
9. Machine learning: clustering
10. Machine learning: model evaluation

## 💾 Installation
## 🛄 Load data
## 🛁 Pre-processing

### Scalers
Rescaling is important because many machine learning algorithms are sensitive to the magnitude and range of a feature. Without scaling, features with a wide range of values will magnify the learning process, and lead to bias results. 

**MinMaxScaler**

Rescales the data such that all values of the feature are in the range from 0 to 1 inclusive. 

```
sklearn.preprocessing.MinMaxScaler
```

**Sample Code**

```
from sklearn.preprocessing import MinMaxScaler

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Check the minimum value
scaler.data_min_

# Check the maximum value
scaler.data_max_
```

<details>
<summary><strong>Manual Code</strong> (optional)</summary>

```
  import numpy as np
  
  data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
  manual = (np.array(data) - scaler.data_min_) / scaler.data_range_
  
  # Check the minimum value
  manual.min(axis=0)
  
  # Check the maximum value
  manual.max(axis=0)
```
</details>

**StandardScaler**

Normalises the feature so the column will have mean = 0, and standard deviation = 1. 

```
sklearn.preprocessing.StandardScaler
```

**Sample Code**

```
from sklearn.preprocessing import StandardScaler

data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Check the mean
scaled_data.mean(axis = 0)

# Check the standard deviation
scaled_data.std(axis = 0)
```

**References**

## 🔩 Feature engineering
## 📐 Dimensionality reduction

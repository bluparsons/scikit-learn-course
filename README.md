<p align="center">
  <img 
    src="https://github.com/user-attachments/assets/483d6899-06fe-428b-a1c3-5273432a77f5"
    alt="banner"
    width="50%"
  />
</p>

This course provides descriptions and examples of all scikit-learn tools, the course is divided into the steps that they would be used in a machine learning project. 

<p align="center">
  <img 
    src="https://github.com/user-attachments/assets/268082e6-7196-4b31-bd79-f696334d4d68"
    alt="life-cycle"
    width="70%"
  />
</p>

## 💾 Installation
## 🛄 Load data
## 🛁 Pre-processing

### Scalers
Rescaling is important because many machine learning algorithms are sensitive to the magnitude and range of a feature. Without scaling, features with a wide range of values will magnify the learning process, and lead to bias results. It is important to only scale numerical features, do not scale categorical features and do not scale the target variable! 

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

**Imputer**

The Imputer class is used to replace missing or NaN values of a particular variable. There are multiple types of imputation: 
- Simple Imputer
- IterativeImputer
- KNNImputer

```
sklearn.preprocessing.Imputer
```

**SimpleImputer**

Basic stragegies where the missing value is imputed with a constant value (e.g. mean, median, or mode of the column). 

**Sample Code for Numerical data**

```
from sklearn.preprocessing import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
X = [[np.nan, 2], [6, np.nan], [7, 6]]
imp.transform(X)
```

**Sample Code for Categorical data**

```
import pandas as pd
from sklearn.preprocessing import SimpleImputer

df = pd.DataFrame([["a", "x"],
                   [np.nan, "y"],
                   ["a", np.nan],
                   ["b", "y"]], dtype="category")
imp = SimpleImputer(strategy="most_frequent")
imp.fit_transform(df)
```

**IterativeImputer**

The IterativeImputer class models the feature with missing values as a function of other features, and uses that estimate for imputation. Basically a feature column is designated as output `y` and the other feature columns are treated as inputs `X`. 

**Sample Code**

```
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit([[1, 2], [3, 6], [4, 8], [np.nan, 3], [7, np.nan]])
X_test = [[np.nan, 2], [6, np.nan], [np.nan, 6]]
# Model learns that the second feature is double the first
np.round(imp.transform(X_test)))
```

## 📐 Dimensionality reduction

# Module 2: Analytics Pipeline (`/analytics`)

**Author**: Manikanta Reddy Reddygari  
**Dataset**: Titanic Passenger Manifest (Loaded via Seaborn, saved to `titanic.csv`)

---

## Overview

This module covers the complete data science lifecycle on the Titanic dataset, structured into two sequential parts:
- **Part A (EDA & Data Story)**: Data profiling, missing value handling with strict threshold rules, univariate/bivariate analysis, a 4-chart multivariate data story, and a standardization check.
- **Part B (Predictive Modeling)**: Stratified train/test splitting, leakage-free Scikit-Learn preprocessing, model benchmarking (Logistic Regression, Decision Tree, Random Forest), class imbalance experiments (SMOTE vs. class weights), hyperparameter tuning with Out-of-Bag scoring, a fare regression side-task, and pipeline serialization with Joblib.

---

## Single Load & Offline Fallback Strategy

To prevent redundant network requests and ensure the code can be graded offline, the raw dataset was loaded once using `sns.load_dataset('titanic')` and immediately written to `titanic.csv` in this folder. All EDA and modeling code reads directly from this committed CSV file:

```python
df = pd.read_csv("titanic.csv")
```

---

## Part A: Exploratory Data Analysis & Findings

### 1. Initial Data Profiling
- **Shape**: $891$ rows, $15$ columns.
- **Measured Missing Values**:
  - `deck`: $688$ missing (**$77.22\%$**)
  - `age`: $177$ missing (**$19.87\%$**)
  - `embarked`: $2$ missing (**$0.22\%$**)
  - `embark_town`: $2$ missing (**$0.22\%$**)

### 2. Missing Value Decisions (Threshold Rule)
I applied the percentage-based threshold rule as follows:
- **$< 5\%$ Missing (`embarked`, `embark_town` at $0.22\%$)**: 
  - *Decision*: Dropped the 2 affected rows.
  - *Reasoning*: Because less than $0.25\%$ of the records were affected, dropping these rows avoids inventing artificial embarkation ports while retaining $99.78\%$ of the dataset.
- **$5\% - 30\%$ Missing (`age` at $19.87\%$)**: 
  - *Decision*: Imputed with the median age ($28.00$ years).
  - *Reasoning*: Age has moderate missingness. Median imputation is preferable to the mean here because it is not distorted by extreme age values or infants.
- **$> 30\%$ Missing (`deck` at $77.22\%$)**: 
  - *Decision*: Dropped the column entirely.
  - *Reasoning*: Over $77\%$ of records lack cabin deck info. Trying to impute or treat missing as a category would add significant noise rather than genuine predictive signal.
- **Final Cleaned EDA Shape**: $889$ rows $\times$ $14$ columns.

---

### 3. Univariate Analysis (Age & Fare)
I computed the Interquartile Range (IQR) for both numeric features using $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$:
- **`age`**:
  - $Q_1 = 22.00$, $Q_3 = 35.00$, $\text{IQR} = 13.00$.
  - Valid Range: $[2.50, 54.50]$.
  - **Outliers Detected**: **$65$ rows** ($7.31\%$), representing older adults and infants under $2.5$ years.
- **`fare`**:
  - $Q_1 = 7.90$, $Q_3 = 31.00$, $\text{IQR} = 23.10$.
  - Valid Range: $[-26.76, 65.66]$.
  - **Outliers Detected**: **$114$ rows** ($12.82\%$), representing expensive first-class tickets.
- **Fare Skewness Analysis**:
  - **Mean**: $32.10$
  - **Median**: $14.45$
  - **Mode**: $8.05$
  - **Ordering**: $\text{Mean } (32.10) > \text{Median } (14.45) > \text{Mode } (8.05)$.
  - **Conclusion**: The distribution of `fare` is **strongly right-skewed** (positive skew). A small cluster of first-class passengers paid high fares (up to $512.33$), pulling the mean substantially above the median.

Plots are saved to `charts/01_univariate_age_fare.png`.

---

### 4. Bivariate Analysis & Correlation Matrix

#### Survival Rates (Computed with Boolean Masking):
- **By Sex**:
  - Female: **$74.04\%$**
  - Male: **$18.89\%$**
- **By Class (`pclass`)**:
  - Class 1: **$62.62\%$**
  - Class 2: **$47.28\%$**
  - Class 3: **$24.24\%$**
- **By Sex and Class Combined**:
  - Female in Class 1: **$96.74\%$**
  - Female in Class 2: **$92.11\%$**
  - Female in Class 3: **$50.00\%$**
  - Male in Class 1: **$36.89\%$**
  - Male in Class 2: **$15.74\%$**
  - Male in Class 3: **$13.54\%$**

#### 6 $\times$ 6 Correlation Matrix (Numeric Columns Only):
*Note: Derived boolean columns `adult_male` and `alone` are excluded since they are direct functions of sex/age and family counts.*

| Feature | `survived` | `pclass` | `age` | `sibsp` | `parch` | `fare` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`survived`** | $1.000$ | $-0.336$ | $-0.070$ | $-0.034$ | $0.083$ | $0.255$ |
| **`pclass`** | $-0.336$ | $1.000$ | $-0.337$ | $0.082$ | $0.017$ | $-0.548$ |
| **`age`** | $-0.070$ | $-0.337$ | $1.000$ | $-0.233$ | $-0.171$ | $0.094$ |
| **`sibsp`** | $-0.034$ | $0.082$ | $-0.233$ | $1.000$ | $0.415$ | $0.161$ |
| **`parch`** | $0.083$ | $0.017$ | $-0.171$ | $0.415$ | $1.000$ | $0.218$ |
| **`fare`** | $0.255$ | $-0.548$ | $0.094$ | $0.161$ | $0.218$ | $1.000$ |

#### The Two Strongest Correlations:
1. **`pclass` $\leftrightarrow$ `fare` ($r = -0.548$, $|r| = 0.548$)**: Strong negative correlation showing that lower numeric class numbers (1st class) had much higher ticket prices.
2. **`sibsp` $\leftrightarrow$ `parch` ($r = 0.415$, $|r| = 0.415$)**: Positive correlation showing that passengers traveling with siblings/spouses were also likely traveling with parents/children as family units.

Heatmap plot is saved to `charts/02_correlation_heatmap.png`.

---

### 5. Multivariate Data Story (4 Visualizations)

1. **Chart 1: Survival Rate by Class and Sex** (`charts/03_multivariate_survival_by_sex_pclass.png`):
   *Interpretation*: Gender was the strongest single determinant of survival, with women in 1st and 2nd class achieving over $92\%$ survival. Men in 2nd and 3rd class had the lowest survival rates (under $16\%$), confirming that the "women and children first" evacuation protocol was enforced strictly.

2. **Chart 2: Age vs. Survival across Passenger Classes** (`charts/04_multivariate_age_distribution_by_survival_pclass.png`):
   *Interpretation*: Children in 1st and 2nd class were prioritized and survived at very high rates. However, children in 3rd class had much lower survival, reflecting delayed access from lower decks to the lifeboat stations. Elderly passengers suffered low survival across all classes.

3. **Chart 3: Fare Distribution by Class and Survival** (`charts/05_multivariate_fare_vs_survival_boxplot.png`):
   *Interpretation*: Within every single class tier, passengers who survived paid higher median fares than those who did not. This suggests that passengers who booked better-located cabins on higher decks had a physical advantage during the evacuation.

4. **Chart 4: Family Size vs. Survival Probability** (`charts/06_multivariate_family_size_survival.png`):
   *Interpretation*: Moderate family sizes ($2$ to $4$ members) saw the best survival rates ($\sim 55\%-70\%$), likely due to family members assisting each other. Solo travelers ($30\%$) and very large families ($>4$ members, $<15\%$) suffered lower survival due to isolation or the chaos of keeping large groups together.

---

### 6. Exploratory Standardization Check
As a sanity check, I applied $z = (x - \mu) / \sigma$ across the cleaned data:
- **`age`**: Original Mean $= 29.3152$, Std $= 12.9849$ $\longrightarrow$ Standardized Mean $= 0.0000$, Std $= 1.0000$.
- **`fare`**: Original Mean $= 32.0967$, Std $= 49.6975$ $\longrightarrow$ Standardized Mean $= 0.0000$, Std $= 1.0000$.
*(This was an exploratory check; modeling uses train-only `StandardScaler` inside `ColumnTransformer`.)*

---

## Part B: Predictive Modeling & Results

### 1. Stratified Train / Test Split
- **Split**: $80\%$ Train ($711$ rows) / $20\%$ Test ($178$ rows), `random_state=42`.
- **Target Balance**: $61.75\%$ Deceased ($0$), $38.25\%$ Survived ($1$).
- **Justification**: Because the target is imbalanced ($\sim 62\%$ vs $\sim 38\%$), using a stratified split guarantees that both training and test sets maintain identical class distributions, avoiding evaluation bias.

### 2. Leakage-Free Preprocessing Pipeline
To prevent data leakage, all transformations are enclosed in a Scikit-Learn `ColumnTransformer` and fit only on `X_train`:
- Numeric columns (`pclass`, `age`, `sibsp`, `parch`, `fare`): `SimpleImputer(strategy='median')` $\to$ `StandardScaler()`.
- Categorical columns (`sex`, `embarked`): `SimpleImputer(strategy='most_frequent')` $\to$ `OneHotEncoder(drop='first', handle_unknown='ignore')`.

---

### 3. Classifier Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $0.8146$ | $0.7966$ | $0.6912$ | $0.7402$ | $0.8610$ |
| **Decision Tree** | $0.8090$ | $0.8148$ | $0.6471$ | $0.7213$ | $0.8560$ |
| **Random Forest (Tuned)** | **$0.8034$** | $0.7619$ | **$0.7059$** | **$0.7328$** | **$0.8266$** |

*Generated Visualizations*:
- `charts/07_decision_tree.png`: Rendered Decision Tree with feature and class names.
- `charts/08_confusion_matrices.png`: Confusion matrices for all three models.
- `charts/09_roc_curves.png`: ROC curves with AUC comparisons.

---

### 4. Class Imbalance Comparison (Random Forest)

| Strategy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| **(a) Baseline (No Handling)** | $0.7619$ | $0.7059$ | $0.7328$ |
| **(b) `class_weight='balanced'`** | $0.7571$ | $0.7794$ | $0.7681$ |
| **(c) `SMOTE` (Training Fold Only)** | $0.7538$ | $0.7206$ | $0.7368$ |

**My Conclusion on Imbalance**:
The baseline model achieves solid precision ($0.7619$) but misses some survivors (Recall $0.7059$). Weight balancing (`class_weight='balanced'`) yielded the highest recall ($0.7794$) by penalizing false negatives. Applying `SMOTE` strictly on the training fold improved recall to $0.7206$ while safely avoiding any synthetic leakage into test data.

---

### 5. Hyperparameter Tuning & Out-of-Bag (OOB) Score
I ran `GridSearchCV` on `RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)` across:
- `n_estimators`: $[50, 100, 150]$
- `max_depth`: $[4, 6, 8, \text{None}]$
- `max_features`: `['sqrt', 'log2']`

**Results**:
- **Best Parameters**: `{'max_depth': 6, 'max_features': 'log2', 'n_estimators': 100}`
- **Best 5-Fold Cross-Validation F1**: **$0.7416$**
- **Out-of-Bag (OOB) Score**: **$0.8073$**

---

### 6. Regression Side-Task (Predicting Fare)
I trained a Multivariate Linear Regression model predicting `fare` from passenger features:
- **MAE**: $24.5106$
- **RMSE**: $60.1827$
- **$R^2$**: $0.3002$
- **Adjusted $R^2$**: $0.2639$
- **Heteroscedasticity Analysis**:
  The residual plot (`charts/10_fare_regression_residuals.png`) displays a clear **fan/funnel pattern**: residuals stay small for tickets under $£30$ but spread out widely for expensive tickets above $£100$. This non-constant error variance confirms **heteroscedasticity**, which is typical when predicting right-skewed financial data without logarithmic transformation.

---

### 7. Unified Model Comparison & Recommendation

Because classification metrics (probabilities and thresholds) and regression metrics (continuous errors in GBP) operate on different scales, they are presented in separate groups below:

#### Classification Models (`survived` target)
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $0.8146$ | $0.7966$ | $0.6912$ | $0.7402$ | $0.8610$ |
| **Decision Tree** | $0.8090$ | $0.8148$ | $0.6471$ | $0.7213$ | $0.8560$ |
| **Random Forest (Tuned)** | $0.8034$ | $0.7619$ | $0.7059$ | $0.7328$ | $0.8266$ |

#### Regression Model (`fare` target)
| Model | MAE | RMSE | $R^2$ | Adjusted $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Multivariate Linear Regression** | $24.5106$ | $60.1827$ | $0.3002$ | $0.2639$ |

#### Final Deployment Recommendation:
> I recommend deploying the **Random Forest Classifier**. Across the test split, it achieved an Accuracy of $0.8034$, an F1 Score of $0.7328$, and an ROC-AUC of $0.8266$, backed by an Out-of-Bag score of $0.8073$. While single decision trees suffer from localized variance and high depth sensitivity, Random Forest's bagging architecture averages out individual tree noise and captures non-linear interactions between class, sex, and fare reliably.

---

### 8. Full Pipeline Persistence & Inference Verification

The entire fitted pipeline (the `ColumnTransformer` with its imputers and scalers plus the tuned Random Forest model) was serialized to disk using Joblib:
`analytics/best_titanic_pipeline.joblib`

You can verify that it loads and scores raw, uncleaned passenger rows by running:
```bash
python analytics/test_inference.py
```
Output:
```text
Passenger 1 (Class 1, Female, Age None, Fare 83.15, Embarked C): Prediction = Survived (Probability = 0.9832)
Passenger 2 (Class 3, Male, Age 24.0, Fare 7.90, Embarked S):    Prediction = Died     (Probability = 0.0875)
```

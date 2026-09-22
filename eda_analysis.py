"""
Exploratory Data Analysis (EDA) on the Titanic Dataset
Author: Data Science Intern
Description: Complete automated script for structural overview, missing values, 
summary statistics, visualizations, and insights.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("=" * 60)
print("1. LOADING DATASET...")
print("=" * 60)
# Load Titanic dataset from Seaborn (Real dataset with 891 rows and mixed columns)
df = sns.load_dataset("titanic")
print("Data loaded successfully!\n")

print("=" * 60)
print("2. STRUCTURAL OVERVIEW")
print("=" * 60)
print(f"Dataset Shape: {df.shape[0]} rows and {df.shape[1]} columns\n")

print("--- Data Types & Info ---")
print(df.info())

print("\n--- Memory Usage ---")
print(df.memory_usage(deep=True))

print("\n--- First 5 Rows ---")
print(df.head())

print("=" * 60)
print("3. MISSING VALUE ANALYSIS")
print("=" * 60)
missing_count = df.isnull().sum()
missing_percentage = (missing_count / len(df)) * 100

missing_df = pd.DataFrame(
    {"Missing Count": missing_count, "Missing Percentage (%)": missing_percentage}
)
print(missing_df[missing_df["Missing Count"] > 0])

# Visualizing missing values using a heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
plt.title("Missing Values Heatmap", fontsize=14, fontweight="bold")
plt.show()
print(
    "Interpretation: 'deck' has massive missing data, while 'age' has moderate missing values.\n"
)

print("=" * 60)
print("4. SUMMARY STATISTICS")
print("=" * 60)
print("--- Numeric Summary Statistics ---")
print(df.describe())

print("\n--- Categorical Value Counts ---")
for col in ["class", "sex", "embark_town", "alive"]:
    print(f"\nValue counts for '{col}':")
    print(df[col].value_counts())

print("=" * 60)
print("5. GENERATING VISUALIZATIONS & INTERPRETATIONS")
print("=" * 60)

# Chart 1: Survival Rate by Passenger Class
plt.figure()
sns.barplot(data=df, x="class", y="survived", palette="Blues_d", errorbar=None)
plt.title("Survival Rate by Passenger Class", fontsize=14, fontweight="bold")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()
print(
    "[Chart 1 Interpretation]: First-class passengers had a significantly higher survival rate compared to second and third-class passengers.\n"
)

# Chart 2: Age Distribution by Survival Outcome
plt.figure()
sns.histplot(
    data=df, x="age", hue="survived", multiple="stack", palette="Set2", bins=30
)
plt.title("Age Distribution by Survival Outcome", fontsize=14, fontweight="bold")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
print(
    "[Chart 2 Interpretation]: Children (ages 0-10) had a higher proportion of survival relative to their count, whereas adults aged 20-40 experienced higher fatalities.\n"
)

# Chart 3: Fare Distribution Across Classes
plt.figure()
sns.boxplot(data=df, x="class", y="fare", palette="Set3")
plt.title("Fare Distribution Across Passenger Classes", fontsize=14, fontweight="bold")
plt.xlabel("Passenger Class")
plt.ylabel("Fare Paid")
plt.ylim(0, 300)
plt.show()
print(
    "[Chart 3 Interpretation]: First-class tickets had a much higher spread of fares paid with high-value outliers, while third-class fares were uniformly cheap.\n"
)

# Chart 4: Survival Rate by Class and Gender
plt.figure()
sns.catplot(
    data=df,
    x="class",
    y="survived",
    hue="sex",
    kind="bar",
    palette="muted",
    height=5,
    aspect=1.5,
    errorbar=None,
)
plt.title("Survival Rate by Class and Gender", fontsize=14, fontweight="bold")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()
print(
    "[Chart 4 Interpretation]: Across all ticket classes, females had a vastly higher survival rate than males due to rescue protocols.\n"
)

# Chart 5: Correlation Matrix
plt.figure()
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(
    correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
)
plt.title("Correlation Matrix of Numeric Features", fontsize=14, fontweight="bold")
plt.show()
print(
    "[Chart 5 Interpretation]: 'fare' shows a positive correlation with survival, while family size variables show a mild negative correlation.\n"
)

print("=" * 60)
print("6. KEY FINDINGS SUMMARY")
print("=" * 60)
print(
    """
1. Socioeconomic Privilege: Passenger class was a strong determinant of survival; 1st-class passengers had priority access.
2. Gender Disparity: Females had exceptionally high survival rates across all classes.
3. Age Vulnerability: Infants and young children were prioritized during rescue operations.
4. Ticket Cost Impact: Higher ticket fares were closely associated with better cabin locations and survival outcomes.
"""
)
print("=" * 60)
print("EDA SCRIPT COMPLETED SUCCESSFULLY!")
print("=" * 60)
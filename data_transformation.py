"""
Experiment 5: Data Transformation and Normalization
Tasks:
  1. Create new calculated columns
  2. Apply normalization and scaling techniques
  3. Rename and modify columns
  4. Convert categorical values where required
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder


def create_sample_dataset() -> pd.DataFrame:
    """Create a sample employee dataset for demonstration."""
    data = {
        'emp_id': [101, 102, 103, 104, 105, 106, 107, 108],
        'Emp_Name': [' alice smith ', 'bob johnson', 'CHARLIE BROWN', '  Diana Prince', 
                     'Evan Wright', 'Fiona Gallagher', 'George Clark', 'Hannah Abbott'],
        'dept': ['IT', 'HR', 'Finance', 'IT', 'Marketing', 'Finance', 'HR', 'IT'],
        'education_level': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'Bachelor', 'High School', 'PhD'],
        'age': ['25', '34', '45', '29', '40', '26', '38', '50'],  # stored as string to demonstrate type modification
        'base_salary': [50000, 75000, 120000, 62000, 95000, 52000, 68000, 140000],
        'bonus_pct': [10, 12, 15, 8, 14, 10, 11, 20],
        'experience_years': [2, 7, 15, 4, 12, 2, 9, 20],
        'remote_status': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No']
    }
    return pd.DataFrame(data)


def print_section(title: str):
    """Helper function to print formatted section headers."""
    print("\n" + "=" * 80)
    print(f" {title} ")
    print("=" * 80)


def main():
    print_section("ORIGINAL DATASET")
    df = create_sample_dataset()
    print(df)
    print("\nData Types:")
    print(df.dtypes)

    # -------------------------------------------------------------------------
    # TASK 1: RENAME AND MODIFY COLUMNS
    # -------------------------------------------------------------------------
    print_section("TASK 1: RENAME AND MODIFY COLUMNS")

    # 1.1 Rename columns for standardized naming convention (clean snake_case)
    rename_mapping = {
        'emp_id': 'employee_id',
        'Emp_Name': 'employee_name',
        'dept': 'department',
        'bonus_pct': 'bonus_percentage'
    }
    df.rename(columns=rename_mapping, inplace=True)
    print("1. Renamed columns (emp_id -> employee_id, Emp_Name -> employee_name, etc.):")
    print(df.columns.tolist())

    # 1.2 Modify column data types: age string -> integer
    df['age'] = pd.to_numeric(df['age'])
    print("\n2. Modified data type of 'age' column to int64:")
    print(f"   'age' dtype: {df['age'].dtype}")

    # 1.3 Modify text data values: strip whitespaces and convert to title case
    df['employee_name'] = df['employee_name'].str.strip().str.title()
    print("\n3. Cleaned and formatted 'employee_name' values:")
    print(df[['employee_id', 'employee_name']])

    # -------------------------------------------------------------------------
    # TASK 2: CREATE NEW CALCULATED COLUMNS
    # -------------------------------------------------------------------------
    print_section("TASK 2: CREATE NEW CALCULATED COLUMNS")

    # 2.1 Calculate monetary bonus amount from percentage
    df['bonus_amount'] = (df['base_salary'] * (df['bonus_percentage'] / 100)).round(2)

    # 2.2 Calculate total compensation (base salary + bonus)
    df['total_compensation'] = df['base_salary'] + df['bonus_amount']

    # 2.3 Calculate salary per year of experience (ratio column)
    df['salary_per_exp_year'] = (df['base_salary'] / df['experience_years']).round(2)

    # 2.4 Create a categorized column based on salary thresholds
    # High: >= 100k, Medium: 60k - 100k, Entry: < 60k
    df['salary_tier'] = pd.cut(
        df['total_compensation'],
        bins=[-np.inf, 60000, 100000, np.inf],
        labels=['Entry Level', 'Mid Level', 'Senior Level']
    )

    print("Created calculated columns: ['bonus_amount', 'total_compensation', 'salary_per_exp_year', 'salary_tier']")
    print(df[['employee_name', 'base_salary', 'bonus_amount', 'total_compensation', 'salary_per_exp_year', 'salary_tier']])

    # -------------------------------------------------------------------------
    # TASK 3: CONVERT CATEGORICAL VALUES WHERE REQUIRED
    # -------------------------------------------------------------------------
    print_section("TASK 3: CONVERT CATEGORICAL VALUES")

    # 3.1 Binary Categorical Mapping: remote_status ('Yes'/'No' -> 1/0)
    df['remote_encoded'] = df['remote_status'].map({'Yes': 1, 'No': 0})
    print("1. Binary Encoding for 'remote_status' (Yes -> 1, No -> 0):")
    print(df[['remote_status', 'remote_encoded']])

    # 3.2 Ordinal Encoding: education_level (meaningful order: High School < Bachelor < Master < PhD)
    education_order = {
        'High School': 0,
        'Bachelor': 1,
        'Master': 2,
        'PhD': 3
    }
    df['education_encoded'] = df['education_level'].map(education_order)
    print("\n2. Ordinal Encoding for 'education_level' (Custom order mapping):")
    print(df[['education_level', 'education_encoded']])

    # 3.3 Label Encoding: alternative method for categorical variables
    le = LabelEncoder()
    df['dept_label_encoded'] = le.fit_transform(df['department'])
    print("\n3. Label Encoding for 'department' using Scikit-Learn:")
    print(df[['department', 'dept_label_encoded']])

    # 3.4 One-Hot Encoding (Nominal): department (no natural ordering)
    # Using pd.get_dummies()
    one_hot_dept = pd.get_dummies(df['department'], prefix='dept', dtype=int)
    df_with_dummies = pd.concat([df, one_hot_dept], axis=1)
    print("\n4. One-Hot Encoding for 'department' (pd.get_dummies):")
    print(one_hot_dept)

    # -------------------------------------------------------------------------
    # TASK 4: APPLY NORMALIZATION AND SCALING TECHNIQUES
    # -------------------------------------------------------------------------
    print_section("TASK 4: APPLY NORMALIZATION AND SCALING TECHNIQUES")

    # 4.1 Min-Max Normalization (Scales values to [0, 1] range)
    # Formula: (X - X_min) / (X_max - X_min)
    # Manual calculation with Pandas:
    df['age_minmax_pandas'] = (
        (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
    ).round(4)

    # Using Scikit-Learn MinMaxScaler:
    min_max_scaler = MinMaxScaler()
    df['base_salary_minmax'] = min_max_scaler.fit_transform(df[['base_salary']]).round(4)
    df['total_comp_minmax'] = min_max_scaler.fit_transform(df[['total_compensation']]).round(4)

    print("1. Min-Max Normalization (Rescales feature values into [0, 1]):")
    print(df[['employee_name', 'base_salary', 'base_salary_minmax', 'total_compensation', 'total_comp_minmax', 'age', 'age_minmax_pandas']])

    # 4.2 Standardization (Z-score Scaling: Mean = 0, Std = 1)
    # Formula: (X - Mean) / Std
    # Manual calculation with Pandas:
    df['age_zscore_pandas'] = (
        (df['age'] - df['age'].mean()) / df['age'].std()
    ).round(4)

    # Using Scikit-Learn StandardScaler:
    standard_scaler = StandardScaler()
    df['base_salary_zscore'] = standard_scaler.fit_transform(df[['base_salary']]).round(4)
    df['total_comp_zscore'] = standard_scaler.fit_transform(df[['total_compensation']]).round(4)

    print("\n2. Z-Score Standardization (Centers mean at 0 with unit variance):")
    print(df[['employee_name', 'base_salary', 'base_salary_zscore', 'total_compensation', 'total_comp_zscore', 'age', 'age_zscore_pandas']])

    # 4.3 Log Transformation (Handling skewed data / variance stabilization)
    df['salary_log_transformed'] = np.log(df['base_salary']).round(4)
    print("\n3. Log Transformation (np.log for reducing variance/skewness):")
    print(df[['employee_name', 'base_salary', 'salary_log_transformed']])

    # -------------------------------------------------------------------------
    # SUMMARY OF FINAL TRANSFORMED DATASET
    # -------------------------------------------------------------------------
    print_section("FINAL TRANSFORMED DATASET (Selected Columns)")
    summary_cols = [
        'employee_id', 'employee_name', 'department', 'age', 
        'total_compensation', 'salary_tier', 'education_encoded', 
        'remote_encoded', 'total_comp_minmax', 'total_comp_zscore'
    ]
    print(df[summary_cols].to_string(index=False))

    # Save to CSV for reference
    output_filename = "transformed_dataset.csv"
    df.to_csv(output_filename, index=False)
    print(f"\n[INFO] Full transformed dataset successfully exported to '{output_filename}'")


if __name__ == '__main__':
    main()

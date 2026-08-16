# Prompt for taking on "eda" tasks
EDA_PROMPT = """
The current task is about exploratory data analysis, please note the following:
- Distinguish column types with `select_dtypes` for tailored analysis and visualization, such as correlation.
- Remember to `import numpy as np` before using Numpy functions.
"""

# Prompt for taking on "data_preprocess" tasks
DATA_PREPROCESS_PROMPT = """
The current task is about data preprocessing, please note the following:
- Monitor data types per column, applying appropriate methods ONLY WHEN NECESSARY OR REQUIRED.
- For all selected columns that are of string types, MAKE SURE to transform them into dummy or categorical variables. BE SURE the new columns should have the same names as their previous names. 
- For all selected columns that are of numerical types (not categorical types), such as int8 or float8, transform them into float64. Some numerical types may ve very dangerous, for example, taking squared value of int8 type number might easily make the result exceed the allowed numerical value range.
- Ensure operations are on existing dataset columns.
- MAKE SURE to figure out whether any column(s) in the dataset is the index (or double index for panel data analysis) of the dataset, and properly set the index accordingly. DO NOT INCORRECTLY TREAT INDEX COLUMNS AS DATA COLUMNS!!! If there is such index column(s), properly set the index into the pd.Series or pd.DataFrame dataset, and then DROP THE ORIGINAL INDEX COLUMN!
- Avoid writing processed data to files.
- Avoid any change to label column, such as standardization, etc.
- Prefer one-hot encoding for categorical data, but DO MAKE SURE the one-hot encoding generated columns are dummy variables of type int32. KEEP IN MIND ALL FINAL DATASETS SHOULD BE NUMERICAL VALUES, and STRING & BOOL CANNOT BE ACCEPTED.
- Always copy the DataFrame before processing it and use the copy to process.
"""

# Prompt for taking on "econometric_algorithm" tasks
ECONOMETRIC_ALGORITHM_PROMPT = """
The current task is about matching and applying an econometric algorithm tool. please note the following:
- ALWAYS APPLY AVAILABLE TOOLS FIRST. Find the tool that most satisfy the target.
- PAY VERY MUCH ATTENTION TO USER TASK REQUIREMENTS! MAKE CLEAR AND MATCH VERY ACCURATELY what the dependent variable, treatment variable, control variables (if any), the instructed econometric algorithm and other requirements are. Strict follow the real instructions and do not "ASSUME" any setting about data or methodology in the task!
- Many tools have the input "target_type" that allows users to denote what the tool functions should output. PAY ATTENTION TO THIS INPUT PARAMETER! When you want to observe the detailed result from the econometric algorithm model, BE SURE TO INPUT "final_model" or "final_models" IN THIS PARAMETER!
- Only when no tools are proper for the target should you initiate to find a model. 
- Use the data from previous task result directly, do not mock or reload data yourself.
- Some results are not quantities and may contain graphs and/or tables. STRICTLY follow the user's instruction and provide possible qualitative conclusions and/or suggestions.
- The final result for each task should NEVER be None or np.nan! If such result appears, GO BACK TO CHECK THE INPUTS AND OUTPUTS OF THE TOOL SELECTED!
"""
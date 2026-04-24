import pandas as pd
import numpy as np


def missing_values(df):
    '''
      Calculates the count and percentage of missing values for each column in a DataFrame.

      Args:
          df (pd.DataFrame): The input pandas DataFrame.

      Returns:
          pd.DataFrame: A DataFrame with two columns: 'Missing Values' (count)
                        and 'Percentage (%)' (percentage of missing values).
    '''
    missing_count=df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_data = pd.DataFrame({
        'Missing Values': missing_count,
        'Percentage (%)': missing_percentage
    })
    return missing_data


#Outlier detection using IQR Method
def data_stats(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Handle non-numeric columns gracefully
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        count = df[column_name].count()
        unique_count = df[column_name].nunique()
        min_val = df[column_name].min()
        max_val = df[column_name].max()
        # For non-numeric columns, mean, std, percentiles are not defined
        mean_val = np.nan
        std_val = np.nan
        left_whisker = np.nan
        percentile_25 = np.nan
        percentile_50 = np.nan
        percentile_75 = np.nan
        right_whisker = np.nan
        left_outer_cnt=np.nan
        right_outer_cnt=np.nan
        total_outliers_per=np.nan
        return (count, unique_count, min_val, max_val, mean_val, std_val, left_whisker, percentile_25, percentile_50, percentile_75, right_whisker,total_outliers_per)

    # Calculate descriptive statistics
    count = df[column_name].count()
    unique_count = df[column_name].nunique()
    min_val = df[column_name].min()
    max_val = df[column_name].max()
    mean_val = df[column_name].mean()
    std_val = df[column_name].std()

    # Calculate percentiles and whiskers
    q1 = df[column_name].quantile(0.25)
    median=df[column_name].median()
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    left_whisker = q1 - 1.5 * iqr
    if min_val>=0 and  left_whisker<0:   #for positive value point left wisker cannot be -ve
      left_whisker = 0
    right_whisker = q3 + 1.5 * iqr

    #left_outer_cnt=df[df[column_name]right_whisker][column_name].count()
    left_outer_cnt=np.sum(df[column_name] < left_whisker)
    right_outer_cnt=np.sum(df[column_name] > right_whisker)
    total_outliers_per=((left_outer_cnt+right_outer_cnt)*100)/count

    return (count, unique_count, min_val, max_val, mean_val, std_val, left_whisker, q1, median, q3, right_whisker,left_outer_cnt,right_outer_cnt,total_outliers_per)
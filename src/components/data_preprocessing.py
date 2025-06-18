import pandas as pd
import calendar
import numpy as np

def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses transaction data for fraud detection analysis.
    
    Args:
        df: Raw transaction DataFrame
        
    Returns:
        Preprocessed DataFrame with engineered features
    """
    
    # Helper functions
    def create_amount_bucket(x: float) -> str:
        """Categorize transaction amounts into buckets"""
        if x <= 5.00:
            return "less than 5 dollar"
        elif 5.00 < x <= 10.00:
            return "b/w 5 to 10 dollar"
        elif 10.00 < x <= 40.00:
            return "b/w 10 to 40 dollar"
        elif 40.00 < x <= 60.00:
            return "b/w 40 to 60 dollar"
        elif 60.00 < x <= 80.00:
            return "b/w 60 to 80 dollar"
        elif 80.00 < x <= 150.00:
            return "b/w 80 to 150 dollar"
        else:
            return "more than 150 dollar"
    
    def city_pop_cat(x: float) -> str:
        """Categorize city population"""
        if x <= 1000.00:
            return "Low_pop"
        elif 1000.00 < x <= 10000.00:
            return "Medium_pop"
        else:
            return "High_pop"
    
    def age_bkt(x: int) -> str:
        """Categorize customer age"""
        if x <= 25:
            return "less than 25"
        elif 25 < x <= 40:
            return "b/w 25 to 40"
        elif 40 < x <= 60:
            return "b/w 40 to 60"
        else:
            return "more than 60"
    
    # 1. Amount bucketing
    df["amount_bkt"] = df["amt"].apply(create_amount_bucket)
    
    # 2. Geographical features
    df['latitudinal_distance'] = abs(round(df['merch_lat'] - df['lat'], 3))
    df['longitudinal_distance'] = abs(round(df['merch_long'] - df['long'], 3))
    
    # 3. Population categorization
    df["population_bkt"] = df["city_pop"].apply(city_pop_cat)
    
    # 4. Date/time features
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['trans_date'] = df['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
    df['trans_date'] = pd.to_datetime(df['trans_date'])
    
    # 5. Age calculation
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_date'] - df['dob']).dt.days / 365.25
    df['age'] = df['age'].round(0).astype(int)
    df["age_bkt"] = df["age"].apply(age_bkt)
    
    # 6. Time features
    df['trans_month'] = df['trans_date_trans_time'].dt.month
    df['Month_name'] = df['trans_month'].apply(lambda x: calendar.month_abbr[x])
    df['transaction_time'] = df['trans_date_trans_time'].dt.time
    
    # 7. Time buckets
    bins = [0, 6, 12, 18, 24]
    labels = ['12AM-6AM', '6AM-12PM', '12PM-6PM', '6PM-12AM']
    df['time_bucket'] = pd.cut(
        df['trans_date_trans_time'].dt.hour,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True
    )
    
    # 8. Gender encoding
    df["gender_encod"] = df["gender"].apply(lambda x: 1 if x == "M" else 0)
    
    # 9. Select final columns
    final_columns = [
        "trans_num", "trans_date", "time_bucket", "cc_num", "amount_bkt",
        "category", "gender", "state", "latitudinal_distance",
        "longitudinal_distance", "population_bkt", "age", "age_bkt",
        "gender_encod", "is_fraud"
    ]
    
    return df[final_columns]
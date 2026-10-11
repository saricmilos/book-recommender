# Entire Cleaning and Feature Engineering

import pandas as pd
import numpy as np
from src.preprocess_location import preprocess_location

# Default country mapping
DEFAULT_COUNTRY_MAPPING = {
    # USA variants
    'usa': 'usa', 'u.s.a.': 'usa', 'us': 'usa', 'america': 'usa', 'u.s.a': 'usa',
    'united states': 'usa', 'united states of america': 'usa', 'united state': 'usa', 
    'united statea': 'usa', 'u.s. of a.': 'usa', 'u.s>': 'usa', 'uusa': 'usa',
    'usa now': 'usa', 'good old usa !': 'usa', 'good old u.s.a.': 'usa',
    'usa (currently living in england)': 'usa', 'usa"': 'usa', 'us virgin islands': 'usa',
    'american samoa': 'usa', 'ca': 'usa', 'nyc': 'usa', 'fl': 'usa', 'tx': 'usa',
    'arizona': 'usa', 'california': 'usa', 'new york': 'usa', 'massachusetts': 'usa',
    'ohio': 'usa', 'colorado': 'usa', 'michigan': 'usa', 'virginia': 'usa',
    'washington': 'usa', 'missouri': 'usa', 'pennsylvania': 'usa', 'illinois': 'usa',
    'nevada': 'usa', 'florida': 'usa', 'north carolina': 'usa', 'south carolina': 'usa',
    'west virginia': 'usa', 'maine': 'usa', 'minnesota': 'usa', 'montana': 'usa',
    'new jersey': 'usa', 'hawaii': 'usa', 'alaska': 'usa', 'texas': 'usa',
    'louisiana': 'usa', 'oh': 'usa', 'nj': 'usa', 'ny': 'usa', 'va': 'usa',
    'pa': 'usa', 'mi': 'usa', 'anystate': 'usa', 'everywhere and anywhere': 'usa', 
    'land of the free': 'usa',
    # UK variants
    'uk': 'united kingdom', 'u.k.': 'united kingdom', 'england': 'united kingdom',
    'england uk': 'united kingdom', 'united kingdom': 'united kingdom', 'scotland': 'united kingdom',
    'wales': 'united kingdom', 'northern ireland': 'united kingdom',
    # Germany variants
    'germany': 'germany', 'deutschland': 'germany', 'germay': 'germany', 'deutsches reich': 'germany',
    'baden-wuerttemberg': 'germany', 'baden-württemberg': 'germany', 'hessen': 'germany',
    'rheinland-pfalz': 'germany', 'bayern': 'germany', 'berlin': 'germany',
    # Spain variants
    'spain': 'spain', 'españa': 'spain', 'espana': 'spain', 'espã±a': 'spain', 
    'andalucia': 'spain', 'catalunya': 'spain', 'catalonia': 'spain', 'pais vasco': 'spain',
    'aragon': 'spain',
    # Italy variants
    'italy': 'italy', 'italia': 'italy', 'l`italia': 'italy', 'italien': 'italy',
    'emilia romagna': 'italy', 'lazio': 'italy', 'sicilia': 'italy', 'veneto': 'italy',
    'toscana': 'italy', 'piemonte': 'italy', 'roma': 'italy', 'milano': 'italy',
    # France variants
    'france': 'france', 'la france': 'france', 'ile de france': 'france',
    'bourgogne': 'france', 'alsace': 'france',
    # Portugal variants
    'portugal': 'portugal', 'alentejo': 'portugal', 'lisboa': 'portugal', 'porto': 'portugal',
    'coimbra': 'portugal', 'azores': 'portugal',
    # China variants
    'china': 'china', 'p.r.china': 'china', 'p.r. china': 'china', 'people`s republic of china': 'china',
    'cn': 'china', 'beijing': 'china', 'shanghai': 'china', 'liaoning': 'china',
    # Australia variants
    'australia': 'australia', 'new south wales': 'australia', 'nsw': 'australia', 'victoria': 'australia',
    'queensland': 'australia', 'tasmania': 'australia', 'canberra': 'australia',
    # India variants
    'india': 'india', 'maharashtra': 'india', 'tamil nadu': 'india', 'punjab': 'india',
    # Misc / unknown
    'n/a': 'unknown', 'none': 'unknown', 'unknown': 'unknown', '"': 'unknown', '-': 'unknown', '.': 'unknown', '*': 'unknown'
}

# Default region mapping
DEFAULT_REGION_MAPPING = {
    'usa': 'North America', 'canada': 'North America', 'mexico': 'North America',
    'united kingdom': 'Europe', 'germany': 'Europe', 'spain': 'Europe', 'italy': 'Europe', 'france': 'Europe',
    'china': 'Asia', 'india': 'Asia', 'japan': 'Asia', 'south korea': 'Asia',
    'australia': 'Oceania', 'new zealand': 'Oceania', 'unknown': 'Unknown'
}


def preprocess_books_ratings_users(
    books_df, 
    ratings_df, 
    users_df, 
    country_mapping=DEFAULT_COUNTRY_MAPPING, 
    region_mapping=DEFAULT_REGION_MAPPING, 
    age_bins=None, 
    age_labels=None
):
    """
    Cleans and preprocesses books, ratings, and users datasets, performs feature engineering,
    and standardizes user location and age groups.
    
    Parameters
    ----------
    books_df : pd.DataFrame
        DataFrame containing books information.
    ratings_df : pd.DataFrame
        DataFrame containing user ratings for books.
    users_df : pd.DataFrame
        DataFrame containing user information.
    country_mapping : dict, optional
        Mapping to standardize country names. Default maps variants to canonical country names.
    region_mapping : dict, optional
        Mapping to assign standardized region/continent for each country. Default included.
    age_bins : list of int, optional
        Bin edges for categorizing user ages into groups. Default [0, 18, 25, 35, 50, 105].
    age_labels : list of int, optional
        Labels for age groups corresponding to bins. Default [1, 2, 3, 4, 5].
    
    Returns
    -------
    merged_df : pd.DataFrame
        Preprocessed DataFrame containing merged user, ratings, and book features.
        Includes user/book statistics, user age groups, and standardized location info.
    """
    
    if age_bins is None:
        age_bins = [0, 18, 25, 35, 50, 105]
    if age_labels is None:
        age_labels = [1, 2, 3, 4, 5]
    
    # Clean column names
    for df in [books_df, ratings_df, users_df]:
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace('-', '_')
        )
    
    # Drop image URLs and preprocess locations
    books_df = books_df.drop(columns=["image_url_s","image_url_m","image_url_l"])
    users_df = preprocess_location(users_df, country_mapping, region_mapping, top_cities=50, top_states=50)
    users_df = users_df.drop(columns="location")
    
    # Fix specific book author/publisher issues
    books_df.loc[books_df['isbn'] == '0751352497', 'book_author'] = 'Dorling Kindersley Publishing Staff'
    books_df.loc[books_df['isbn'] == '9627982032', 'book_author'] = 'Credit Suisse'
    books_df.loc[books_df["book_title"]=="Tyrant Moon","publisher"] = "Mundania Press LLC"
    books_df.loc[books_df["book_title"]=="Finders Keepers","publisher"] = "Random House Publishing Group"

    # Fix year_of_publication
    books_df["year_of_publication"] = pd.to_numeric(books_df["year_of_publication"], errors='coerce')
    books_df.loc[(books_df["year_of_publication"] < 1000) | (books_df["year_of_publication"] > 2025), "year_of_publication"] = np.nan
    books_df["year_of_publication"] = books_df["year_of_publication"].fillna(books_df["year_of_publication"].median())
    books_df["year_of_publication"] = books_df["year_of_publication"].astype(int)
    books_df = books_df[books_df['year_of_publication'].between(1900, 2025)]
    
    # Fix user age
    users_df.loc[(users_df['age'] <= 5) | (users_df['age'] > 99), 'age'] = np.nan
    median, std = users_df['age'].median(), users_df['age'].std()
    nulls = users_df['age'].isna().sum()
    random_age = np.clip(np.random.normal(loc=median, scale=std, size=nulls), 5, 100)
    users_df.loc[users_df['age'].isna(), 'age'] = random_age.round().astype(int)
    
    # Merge datasets
    users_ratings = pd.merge(users_df, ratings_df, on='user_id')
    merged_df = pd.merge(users_ratings, books_df, on='isbn')

    merged_df = merged_df[merged_df['book_rating']!=0]
    
    # Feature engineering
    merged_df.loc[:, 'user_avg_rating'] = merged_df.groupby('user_id')['book_rating'].transform('mean')
    merged_df.loc[:, 'user_num_ratings'] = merged_df.groupby('user_id')['book_rating'].transform('count')

    
    merged_df.loc[:, 'book_avg_rating'] = merged_df.groupby('book_title')['book_rating'].transform('mean')
    merged_df.loc[:, 'book_num_ratings'] = merged_df.groupby('book_title')['book_rating'].transform('count')
    merged_df.loc[:, 'book_popularity_score'] = merged_df['book_avg_rating'] * np.log1p(merged_df['book_num_ratings'])
    
    merged_df.loc[:, 'author_avg_rating'] = merged_df.groupby('book_author')['book_rating'].transform('mean')
    merged_df.loc[:, 'publisher_avg_rating'] = merged_df.groupby('publisher')['book_rating'].transform('mean')

    merged_df.loc[:, 'book_age'] = 2025 - merged_df['year_of_publication']
    
    # --- Add User Age Group ---
    merged_df['User_age_Group'] = pd.cut(
        merged_df['age'],
        bins=age_bins,
        labels=age_labels,
        right=False
    ).astype(int)
    
    return merged_df

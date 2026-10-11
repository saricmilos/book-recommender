import pandas as pd

def split_location(df):
    location_split = df['location'].str.split(',', expand=True)
    df['city'] = location_split[0].str.strip()
    df['state'] = location_split[1].str.strip()
    df['country'] = location_split[2].str.strip()
    df[['city', 'state', 'country']] = df[['city', 'state', 'country']].fillna('unknown')
    return df

def clean_country(df, country_mapping, region_mapping):
    df['country'] = df['country'].str.lower().str.strip()
    df['country'] = df['country'].map(lambda x: country_mapping.get(x, x))
    counts = df['country'].value_counts()
    rare_countries = counts[counts <= 10].index
    df['country_clean'] = df['country'].apply(lambda x: 'other' if x in rare_countries else x)
    df['country_clean'] = df['country_clean'].str.strip().str.replace('"', '')
    df['country_clean'] = df['country_clean'].replace('', 'unknown')
    df['region'] = df['country_clean'].map(lambda x: region_mapping.get(x, 'other'))
    df = df.drop(columns=['country'])
    return df

def clean_city(df, top_n=50):
    df['city_clean'] = df['city'].str.strip().str.lower()
    df['city_clean'] = df['city_clean'].replace({'': 'unknown', 'n/a': 'unknown'})
    city_counts = df['city_clean'].value_counts()
    top_cities = city_counts.nlargest(top_n).index
    df['city_clean'] = df['city_clean'].apply(lambda x: x if x in top_cities else 'other')
    df = df.drop(columns=['city'])
    return df

def clean_state(df, top_n=50):
    df['state_clean'] = df['state'].str.strip().str.lower()
    df['state_clean'] = df['state_clean'].replace(['', 'n/a', '\\n/a"', 'na'], 'other')
    top_states = df['state_clean'].value_counts().nlargest(top_n).index
    df['state_clean'] = df['state_clean'].apply(lambda x: x if x in top_states else 'other')
    df = df.drop(columns=['state'])
    return df

def preprocess_location(df, country_mapping, region_mapping, top_cities=50, top_states=50):
    df = split_location(df)
    df = clean_country(df, country_mapping, region_mapping)
    df = clean_city(df, top_cities)
    df = clean_state(df, top_states)
    return df

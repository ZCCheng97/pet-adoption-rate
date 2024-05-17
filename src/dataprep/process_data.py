import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from transformers import pipeline

df = pd.read_csv("data/pets_prepared.csv")
def breed_clean(df):
    """
    * Changed all Breed2 values to 0 for cases where Breed1 == 307 (since there is inconsistent filling of Breed2 as either 0 or 307 in cases where
    Breed1 is 307 already.)
    * Changed all Breed2 values to 0 for cases where BreedPure is "Y" (some pets have the 2 Breed columns filled with the same number which is
    redundant, while some have Breed1 filled only.)
    * Added another column KnownMixed which can be "Pure", "KnownMixed" or "UnknownMixed" depending on whether the 2 breeds are known in an
    animal labeled as "MixedBreed".
    """
    dfc = df.copy()
    dfc["KnownMixed"] = "Pure"
    dfc.loc[(dfc['Breed2'] == 307), 'Breed1'] = 307
    dfc.loc[(dfc['Breed1'] == 307), 'Breed2'] = 0
    dfc.loc[(dfc['BreedPure'] == "Y"), 'Breed2'] = 0
    dfc.loc[(dfc['BreedPure'] == "N") & (dfc['Breed1'] != 307), 'KnownMixed'] = "KnownMixed"
    dfc.loc[(dfc['BreedPure'] == "N") & (dfc['Breed1'] == 307), 'KnownMixed'] = "UnknownMixed"
    return dfc
def full_color(df):
    """
    Adds "FullColor" column that joins the color IDs of the 3 Color columns and treats pet colour as a combination of all 3 colours.
    """
    dfc = df.copy()
    dfc[['Color1', 'Color2', 'Color3']] = dfc[['Color1', 'Color2', 'Color3']].astype(str)
    dfc["FullColor"] = dfc[['Color1', 'Color2', 'Color3']].agg('-'.join, axis=1)
    return dfc
def rescue_rating(df):
    """
    Adds "RescueRating" column and "RescuerID_or_new" column that adds the RescuerID value count and a column that returns rescuer ID if the
    RescueRating is above 5, otherwise is logged as "Threshold_Not_Met". 
    """
    dfc = df.copy()
    z = dfc['RescuerID'].value_counts()
    z1 = z.to_dict()
    dfc['RescueRating'] = dfc['RescuerID'].map(z1) 
    dfc['RescuerID_or_new'] = dfc['RescuerID']
    dfc.loc[(dfc['RescueRating'] <= 5), 'RescuerID_or_new'] = 'Threshold_Not_Met'
    return dfc
def health_uncertainty(df):
    """
    Adds "HealthUncertainty" column that is given "Y" yes value if all 3 Vaccinated, Dewormed, and Sterilised columns are "Not Sure". 
    """
    dfc = df.copy()
    dfc['HealthUncertainty'] = "N"
    dfc.loc[(dfc['VaccinatedName'] == "Not Sure") & (dfc['DewormedName'] == "Not Sure") & (dfc['SterilizedName'] == "Not Sure"),
             'HealthUncertainty'] = 'Y'
    dfc["UncertainRating"] = (df['VaccinatedName'] == "Not Sure").astype(int)+(df['DewormedName'] == "Not Sure").astype(int)+(df['SterilizedName'] == "Not Sure").astype(int)
    return dfc
def desc_length(df):
    """
    Adds "DescLength" column that gives the length of the description.
    """
    dfc = df.copy()
    dfc["DescLength"] = dfc.Description.apply(lambda x: len(str(x).split(" ")))
    return dfc
def text_preprocess(text):
    """
    Preprocesses text before carrying out sentiment analysis. Performs case normalisation and removing any unicode characters.
    """
    import re
    textl = text.lower()
    textlr = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z\!\.\,\? \t])|(\w+:\/\/\S+)|^rt|http.+?", "", textl)
    return textlr
def desc_sentiment(df):
    """
    Add "DescSentiment" column which is a measure of positive/negative sentiment about the description.
    """
    dfc = df.copy()
    # create pipeline for sentiment analysis
    classification = pipeline('sentiment-analysis',max_length=512,
                          truncation=True)
    texts =  dfc.Description.fillna("No Description given.").to_list()
    texts = list(map(text_preprocess,texts))
    sentiments = classification(texts)
    probs = [d['score'] if d['label'].startswith('P') else 1 - d['score'] for d in sentiments]
    dfc["DescSentiment"] = np.array(probs)
    return dfc

df_copy2 = breed_clean(df)
df_copy2 = full_color(df_copy2)
df_copy2 = rescue_rating(df_copy2)
df_copy2 = health_uncertainty(df_copy2)
df_copy2 = desc_length(df_copy2)
df_copy2 = desc_sentiment(df_copy2)

df_copy2.to_csv("data/processed_data.csv")
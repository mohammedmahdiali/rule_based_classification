#############################################
# RULE BASED CLASSIFICATION - تصنيف العملاء
#############################################

# PRICE: صرف العميل
# SOURCE: جهاز العميل
# SEX: جنسية العميل
# COUNTRY: بلد العميل
# AGE: عمر العميل


################# BEFORE - قبل #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17


################# AFTER - بعد #####################

#       customers_level_based        price segment
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C

import numpy as np
import pandas as pd
import seaborn as sns

DATA_PATH = "../Data/persona.csv"
df = pd.read_csv(DATA_PATH)


################# DATA UNDERSTANDING #####################

# Int64Index: 5000 entries, 0 to 4999
# Data columns (total 5 columns):
#  #   Column   Non-Null Count  Dtype 
# ---  ------   --------------  ----- 
#  0   PRICE    5000 non-null   int64 
#  1   SOURCE   5000 non-null   object
#  2   SEX      5000 non-null   object
#  3   COUNTRY  5000 non-null   object
#  4   AGE      5000 non-null   int64

df.columns = ["price", "source", "sex", "country", "age"]

df.price.value_counts()

# 29    1305
# 39    1260
# 49    1031
# 19     992
# 59     212
# 9      200

df.describe()

#              price          age
# count  5000.000000  5000.000000
# mean     34.132000    23.581400
# std      12.464897     8.995908
# min       9.000000    15.000000
# 25%      29.000000    17.000000
# 50%      39.000000    21.000000
# 75%      39.000000    27.000000
# max      59.000000    66.000000

df.source.unique()

# ['android' 'ios']

df.sex.value_counts()

# female    2621
# male      2379

df.country.value_counts()

# usa    2065
# bra    1496
# deu     455
# tur     451
# fra     303
# can     230

df.groupby("country").agg({"price":"sum"}).sort_values(by="price", ascending=False)         

# country  price     
# usa      70225
# bra      51354
# tur      15689
# deu      15485
# fra      10177
# can       7730

df.groupby("country").agg({"price":"mean"}).sort_values(by="price", ascending=False)         
             
# country  price         
# tur      34.787140
# bra      34.327540
# deu      34.032967
# usa      34.007264
# can      33.608696
# fra      33.587459

df.groupby("source").agg({"price":"mean"}).sort_values(by="price", ascending=False)             

# source   price     
# android  34.174849
# ios      34.069102

df.groupby(["country", "source"]).agg({"price":"mean"}).sort_values(by="price", ascending=False)                  

# country source   price     
# tur     android  36.229437
# bra     android  34.387029
# usa     ios      34.371703
# fra     android  34.312500
# deu     ios      34.268817
# bra     ios      34.222222
# can     ios      33.951456
# deu     android  33.869888
# usa     android  33.760357
# can     android  33.330709
# tur     ios      33.272727
# fra     ios      32.776224


################# DATA PROCESSING #####################

agg_df = df.groupby([
        "country", "source", "sex", "age"
    ]).agg({
            "price":"mean"
        }).sort_values(by="price", ascending=False).reset_index()

agg_df["new_age"] = pd.cut(agg_df["age"], bins=[0, 18, 23, 30, 40, 70], labels=['0_18', '19_23', '24_30', '31_40', '41_70'])

#   country   source     sex  age  price new_age
# 0     bra  android    male   46   59.0   41_70
# 1     usa  android    male   36   59.0   31_40
# 2     fra  android  female   24   59.0   24_30
# 3     usa      ios    male   32   54.0   31_40
# 4     deu  android  female   36   49.0   31_40

agg_df["customer_level_based"] = ["_".join(col) for col in agg_df.drop(["age", "price"], axis=1).values]

#   country   source     sex  age  price new_age      customer_level_based
# 0     bra  android    male   46   59.0   41_70    bra_android_male_41_70
# 1     usa  android    male   36   59.0   31_40    usa_android_male_31_40
# 2     fra  android  female   24   59.0   24_30  fra_android_female_24_30
# 3     usa      ios    male   32   54.0   31_40        usa_ios_male_31_40
# 4     deu  android  female   36   49.0   31_40  deu_android_female_31_40

agg_df = agg_df.groupby("customer_level_based").agg({"price":"mean"}).reset_index()

#        customer_level_based  price
# 0   bra_android_female_0_18  35.645303
# 1  bra_android_female_19_23  34.077340
# 2  bra_android_female_24_30  33.863946
# 3  bra_android_female_31_40  34.898326
# 4  bra_android_female_41_70  36.737179

agg_df["segment"] = pd.qcut(agg_df["price"], 4, labels=["D", "C", "B", "A"])

#    customer_level_based      price            segment
# 0  bra_android_female_0_18   35.645303        B
# 1  bra_android_female_19_23  34.077340        C
# 2  bra_android_female_24_30  33.863946        C
# 3  bra_android_female_31_40  34.898326        B
# 4  bra_android_female_41_70  36.737179        A


new_user = 'bra_android_female_31_40'
agg_df[agg_df['customer_level_based'] == new_user]

#        customer_level_based      price segment
# 3  bra_android_female_31_40  34.898326       B


############ Functionalization ############

def rule_based_classifier(dataframe, bins=[0, 18, 23, 30, 40, 70], labels=['0_18', '19_23', '24_30', '31_40', '41_70'], export_to_csv=False):
    
    agg_df = df.groupby([
            "country", "source", "sex", "age"
        ]).agg({
                "price":"mean"
            }).sort_values(by="price", ascending=False).reset_index()

    agg_df["new_age"] = pd.cut(agg_df["age"], bins=bins, labels=labels)
    
    agg_df["customer_level_based"] = ["_".join(col) for col in agg_df.drop(["age", "price"], axis=1).values]
    
    agg_df = agg_df.groupby("customer_level_based").agg({"price":"mean"}).reset_index()
    
    agg_df["segment"] = pd.qcut(agg_df["price"], 4, labels=["D", "C", "B", "A"])
    
    if export_to_csv:
        agg_df.to_csv("rule_based_result.csv")
    
    return agg_df

rule_based = rule_based_classifier(df, export_to_csv=True)
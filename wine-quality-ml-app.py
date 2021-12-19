import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import shap

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

import tensorflow as tf
from tensorflow import keras

data_dir = 'data/train.csv'
WINE_EMOJI_URL = "https://img.charactermap.one/google/android-11/512px/1f377.png"
SPARKLE_EMOJI_URL = "https://img.charactermap.one/google/android-11/512px/2728.png"


# Set page title and favicon.
st.set_page_config(
    page_title="Wine Quality Score Generator", page_icon=WINE_EMOJI_URL,
)

## Main Panel

st.image([SPARKLE_EMOJI_URL, WINE_EMOJI_URL, SPARKLE_EMOJI_URL], width=80)

st.write("""
# Wine Quality Score Generator

Predict the quality of wine based on its physicochemical attributes with this app!

""")

st.caption("by [Ruth G. N.](https://www.linkedin.com/in/ruthgn/)")

"""
[![Follow](https://img.shields.io/twitter/follow/RuthInData?style=social)](https://www.twitter.com/RuthInData)
&nbsp[![Fork](https://img.shields.io/github/forks/ruthgn/Wine-Quality-Prediction-App.svg?logo=github&style=social)](https://github.com/ruthgn/Wine-Quality-Prediction-App)
"""

## Sidebar

st.sidebar.header('Variable Description')
st.sidebar.markdown("""
- **Fixed acidity**: Acids that naturally occur in the grapes used to ferment the wine and carry over into the wine.
- **Volatile acidity**: Mainly acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste.
- **Citric acid**: Acid supplement typically found in small quantites to boost acidity—adding 'freshness' and flavor to wines.
- **Residual sugar**: The amount of sugar remaining after fermentation stops—higher residual sugar makes wine taste sweeter.
- **Chlorides**:  The amount of chloride salts present in the wine.
- **Free sulfur dioxide**: Free form of sulfur dioxide yielded after adding the compound to wine; sulfur dioxide prevents microbial growth and oxidation in wine.
- **Total sulfur dioxide**: Amount of free and bound forms of sulfur dioxide present in wine.
- **Density**: The "thickness" of wine juice depending on the percent alcohol and sugar content.
- **pH**: A measure of the acidity of wine; lower pH means more acidic characteristics.
- **Sulphates**: Amount of potassium sulphate added to wine as antimicrobial and antioxidant agent. 
- **Alcohol**: ABV or percent alcohol content of wine.
""")

st.sidebar.markdown("----")

st.sidebar.header('Background')

st.sidebar.markdown("""
So much about wine making remains elusive. Winemakers, connoisseurs, and scientists have greatly contributed their expertise to the process, but there is still more to be discovered about the art and science of winemaking. 
This project was built to gain a better understanding of what makes a good quality or bad quality wine according to wine experts' taste-buds and mimick their decision-making process in scoring wine quality.
""")
st.sidebar.markdown("----")

st.sidebar.header('The Prediction Model')
st.sidebar.markdown("""
This app runs on a neural network-based model trained on the [Wine Quality Data Set from UCI](https://www.kaggle.com/ruthgn/wine-quality-data-set-red-white-wine).

Kaggle notebook outlining the prediction model building process is available [here](https://www.kaggle.com/ruthgn/predicting-wine-quality-deep-learning-approach).
""")


# Main Panel

st.subheader('Specify Wine Variables')

st.markdown("\n")


### Listing features for user input

st.markdown("***Upload Input CSV File*** (best for bulk entry):")
st.caption("[Example CSV input file](https://github.com/ruthgn/Wine-Quality-Prediction-App/blob/main/data/sample_test.csv)")

#### Setting input parameters (collecting user input)
uploaded_file = st.file_uploader("",type=["csv"])
if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    input_data.index +=1


else:

    test_data_param = pd.read_csv(data_dir) 
    X = test_data_param.drop('quality', axis=1)

    st.markdown("\n")
    st.markdown("or ***Specify Input Parameters***:")
    st.caption(':point_left: Check the sidebar for more details!')
    st.markdown("\n")

    def user_input_features():
        Type = st.selectbox("Type of Wine", ('Red Wine', 'White Wine'))
        FixedAcidity = st.slider("Fixed acidity (g/dm³)", X['fixed acidity'].min(), X['fixed acidity'].max(), float(X['fixed acidity'].median()), step=0.1, format="%0.1f")
        VolatileAcidity = st.slider("Volatile acidity (g/dm³)", X['volatile acidity'].min(), X['volatile acidity'].max(), float(X['volatile acidity'].median()))
        CitricAcid = st.slider("Citric acid (g/dm³)", X['citric acid'].min(), X['citric acid'].max(), float(X['citric acid'].median()))
        ResidualSugar = st.slider("Residual sugar (g/dm³)", X['residual sugar'].min(), X['residual sugar'].max(), float(X['residual sugar'].median()), step=0.1, format="%0.1f")
        Chlorides = st.slider("Chlorides (g/dm³)", X['chlorides'].min(), X['chlorides'].max(), float(X['chlorides'].median()), step=0.001, format="%0.3f")
        FreeSulfurDioxide = st.slider("Free Sulfur Dioxide (mg/dm³)", round(X['free sulfur dioxide'].min()), round(X['free sulfur dioxide'].max()), round(X['free sulfur dioxide'].median()))
        TotalSulfurDioxide = st.slider("Total Sulfur Dioxide (mg/dm³)", round(X['total sulfur dioxide'].min()), round(X['total sulfur dioxide'].max()), round(X['total sulfur dioxide'].median()))
        Density = st.slider("Density (g/cm³)", X['density'].min(), X['density'].max(), float(X['density'].median()), step=0.00001, format="%0.5f")
        pH = st.slider("pH", 2.9, 4.2, float(X['pH'].median()))
        Sulphates = st.slider("Sulphates (g/dm³)", X['sulphates'].min(), X['sulphates'].max(), float(X['sulphates'].median()))
        Alcohol = st.slider("Alcohol (% by volume)", 5.0, 15.0, float(X['alcohol'].median()), step=0.1, format="%0.1f")
        data = {"type": Type.lower().split()[0],
                "fixed acidity": FixedAcidity,
                "volatile acidity": VolatileAcidity,
                "citric acid": CitricAcid,
                "residual sugar": ResidualSugar,
                "chlorides": Chlorides,
                "free sulfur dioxide": FreeSulfurDioxide,
                "total sulfur dioxide": TotalSulfurDioxide,
                "density": Density,
                "pH": pH,
                "sulphates": Sulphates,
                "alcohol": Alcohol}
        features = pd.DataFrame(data, index=[0])
    
        return features

    input_data = user_input_features()
    input_data.index +=1
    

st.markdown("----")
st.write("Input Summary:")
st.table(input_data.style.format({
    "fixed acidity": "{:.1f}",
    "volatile acidity": "{:.2f}",
    "citric acid": "{:.2f}",
    "residual sugar": "{:.1f}",
    "chlorides": "{:.3f}",
    "density": "{:.5f}",
    "pH": "{:.2f}",
    "sulphates": "{:.2f}",
    "alcohol": "{:.1f}"
    }))


st.write('---')


# Loading training and test(input) data

def load_data(input_data):
    ds_train = pd.read_csv(data_dir)
    X_train = ds_train.drop('quality', axis=1)
    X_train = pd.DataFrame(preprocessor.fit_transform(X_train))
    X_train.columns = features_num + ["type_red", "type_white"]

    X_input = input_data
    X_input = pd.DataFrame(preprocessor.transform(X_input))
    X_input.columns = features_num + ["type_red", "type_white"]

    return X_train, X_input

## Data Preprocessing

### Encoding (Categorical Data)
transformer_cat = make_pipeline(
    OneHotEncoder(handle_unknown='ignore'),
)

### Scaling (Numeric Data)
transformer_num = make_pipeline(
    MinMaxScaler(),
)

features_num = [
    'fixed acidity', 'volatile acidity', 'citric acid',
    'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulphates', 
    'alcohol'
]

features_cat = ["type"]


preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat),
)

#Finally load the pre-processed data
_, X_input = load_data(input_data)


# Reads in saved TF Neural Network model
reconstructed_model = tf.keras.models.load_model("my_model")


# Apply model to make predictions
prediction = reconstructed_model.predict(X_input)


## Show predicted wine quality score

score_description = {10:"World Class", 9:"Excellent", 8:"Very Good", 7:"Good", 6:"Above Average", 5:"Average", 4:"Below Average", 3:"Poor", 2:"Very Poor", 1:"Terrible"}

st.header('Wine Score')

if len(list(prediction)) == 1:
    single_prediction = round(prediction[0,0], 1)
    st.subheader(single_prediction)
    if round(single_prediction) >= 9.5:
        st.balloons()
        st.success("This wine is *world class*! (10/10)")
    elif round(single_prediction) >= 8.5:
        st.balloons()
        st.success("This wine is *excellent*! (9/10)")
    elif round(single_prediction) >= 7.5:
        st.balloons()
        st.success("This wine is *very good*. (8/10)")
    elif round(single_prediction) >= 6.5:
        st.info("This wine is *good*. (7/10)")
    elif round(single_prediction) >= 5.5:
        st.info("This wine is *above average*. (6/10)")
    elif round(single_prediction) >= 4.5:
        st.warning("This wine is *average*. (5/10)")
    elif round(single_prediction) >= 3.5:
        st.warning("This wine is *below average*. (4/10)")
    elif round(single_prediction) >= 2.5:
        st.error("This wine is *poor*. (3/10)")
    elif round(single_prediction) >= 1.5:
        st.error("This wine is *very poor*. (2/10)")
    else:
        st.error("This wine is *probably not for drinking*!")

else:
    array_prediction = pd.DataFrame(prediction)
    array_prediction.columns = ['Wine Score']
    array_prediction['Quality'] = array_prediction['Wine Score'].apply(lambda x: score_description[int(round(x))])
    array_prediction.index +=1
    
    st.table(array_prediction.style.format({"Wine Score": "{:.1f}"}))

st.write('---')
st.caption("Ran into a bug or saw something on the app that needs to be improved? Submit a [Github issue](https://github.com/ruthgn/Wine-Quality-Prediction-App/issues) or [fork the repository](https://github.com/ruthgn/Wine-Quality-Prediction-App) to create a pull request.")

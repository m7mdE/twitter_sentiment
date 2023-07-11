### CODING Mini-APP IN STREAMLIT

### import libraries
import pandas as pd
import streamlit as st
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import altair as alt

# custom styling
st.markdown(
        f"""
        <style>
        button.css-b3z5c9.e1ewe7hr10 {{
            background-color: rgb(29, 155, 240);
            padding-left: 16px;
            padding-right: 16px;
            color: #fff;
        }}
        .row-widget.stTextInput.css-pb6fr7.e1q7aese0, .container-result {{
            border: 1px solid #f9f9f900;
            padding: 20px 20px 40px;
            box-shadow: 0px 0px 20px 4px #413f3f52;
            margin: 15px 0;
        }}
        .text-right {{
            float:right;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )

# Title
st.title("Twitter Sentiment")
# Description
st.write("This mini-app predicts a Tweet whether a positive or negative Tweet")

# function for TF-IDF to work properly 
lemmatizer_tfidf = WordNetLemmatizer()
def tokenizer(text):
    return [lemmatizer_tfidf.lemmatize(word) for word in text.split()]

# Loading:
#  TF-IDF Vectorization
#  Robust Scaler
#  Logisitic Regression Model

vectorizer = pickle.load(open('model/tfidf_vectors.pkl', 'rb'))
scaler = pickle.load(open('model/robust_scaled.pkl', 'rb'))
model = pickle.load(open('model/logreg_model.pkl', 'rb'))

# Set up input field
text = st.text_input('What is happening?!', placeholder='Type here...')
# set up a button
clicked = st.button('Tweet')


if clicked:
    if text:

        # Transforming the vectorizer before using
        transform_text = vectorizer.transform([text])
        
        vect_text = pd.DataFrame(columns=vectorizer.get_feature_names_out(), data=transform_text.toarray())

        # Scaling by using Robust Scaler before moving to modeling
        scaled_text = scaler.transform(vect_text)

        # Model Prediction
        prediction = model.predict_proba(scaled_text)

        # Creating a list of class labels
        class_labels = ['Negative Tweet', 'Positive Tweet']
        predict_prob = prediction.flatten()

        # creating a data frame
        data = {'Class Labels': class_labels, 'Predicted Probabilities': predict_prob}
        df = pd.DataFrame(data)
        
        # Classifying the model
        if prediction[:,1] > 0.5:
            st.write(f' "{text}" <br/> <span class="text-right">We predict a <span style="color:#007BFF; font-weight:600">positive Tweet!</span></span>', unsafe_allow_html=True)
        else:
            st.write(f' "{text}" <br/><span class="text-right"> :heavy_exclamation_mark: We predict a <span style="color:#ff0000; font-weight:600">negative Tweet</span> :heavy_exclamation_mark:</span>', unsafe_allow_html=True)

        #######################################################################################################################################
        ### DATA ANALYSIS & VISUALIZATION

        # Displaying Dataframe
        df

        # Specifying color for each class label
        color_scale = alt.Scale(domain=class_labels, range=['#FF0000', '#007BFF'])

        # Creating a barchart
        chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Class Labels', axis=alt.Axis(labels=True, ticks=False, labelAngle=0)),
        y='Predicted Probabilities',
        color=alt.Color('Class Labels', scale=color_scale)
        ).properties(
            width=700,
            height=400,
            title = 'Twitter Probability'
        ).configure_title(
        fontSize=20,
        font='Arial',
        anchor='middle',
        color='white'
        )

        # Render the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write('Please Enter a Tweet')




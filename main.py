import streamlit as st
import data_manipulation as app1
import model_design as app2
import train as app3
import predict as app4

# Define pages based on apps imported.
PAGES = {
    "Data Manipulation": app1,
    "Model Design": app2,
    "Training": app3,
    "Predicting": app4
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()

# st.pyplot(get_common_wordcloud(data_df))

# st.write(data_df)
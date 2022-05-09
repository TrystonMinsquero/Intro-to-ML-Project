import streamlit as st
import data_manipulation as app1
import train as app2
import train as app3

# Define pages based on apps imported.
PAGES = {
    "Data Manipulation": app1,
    "Train": app2,
    "Predict": app3
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()

# st.pyplot(get_common_wordcloud(data_df))

# st.write(data_df)
import streamlit as st
import pandas as pd

#########################################
st.write ('Text Elements')
#########################################

st.markdown('# This markdown has one "#"')
st.markdown('## This markdown has two "#"')
st.markdown('### This markdown has three "#"')
st.title('This is a title')
st.header('This is a header')
st.subheader('This is a subheader')
st.text('This is text')

#########################################
# read using hard-coded filename
#########################################
st.markdown('### Read Dataset')
data = pd.read_csv('./datasets/housing_paml.csv')

#########################################
# read using file upload
#########################################
#data = st.file_uploader("Choose a file")
#if data:
#    data = pd.read_csv('../datasets/housing/housing.csv')

#########################################
# read using cloud-base resource i.e. URL
#########################################
#url = st.text_input('Enter URL')
#st.write('You entered: '+url) # see example in the book

st.write(data)
#st.dataframe(data) # alternative

#########################################
st.markdown('## Columns')
#########################################
col1, col2 = st.columns([1, 1])

with(col1):
    data1 = st.file_uploader("Choose a file")
    if data1:
        data1 = pd.read_csv(data1)
        col1.line_chart(data1['longitude'])
with(col2):
    url = st.text_input('Enter URL')
    st.write('You entered: '+url)

#########################################
st.markdown('## Selectbox options')
#########################################

# Select 1
option = st.selectbox(
    'Select a fetaure',
    options=data.columns)
st.write('You selected:', option)

# Select multiple
options = st.multiselect(
    'Select a fetaure',
    options=data.columns)

st.write('You selected:', options)

#########################################
#st.markdown('## Sidebar/Slider')
#########################################

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )

values = st.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0))
st.write('Values:', values)

#########################################
st.markdown('## Tabs')
#########################################

tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])

with tab1:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

with tab2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

#########################################
st.markdown('## Session State')
#########################################

st.session_state['data'] = data
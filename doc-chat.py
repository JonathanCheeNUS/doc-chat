import streamlit as st
import pandas as pd
import os
import base64
from io import BytesIO
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List

def save_uploaded_file(uploadedfile):
    with open(os.path.join("data", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to data dir. You can now select the file in Step 1".format(uploadedfile.name))

# Variables 
data_folder = 'data'
files = []
mapping_file_to_root = {}
SAMPLE_SIZE = 100
OPEN_API_KEY = 'ENTER API KEY HERE'
# LLM Initialisation and prompt to be used. Can be placed in a separate file as we scale up the services
template = '''
You are a data scientist tasked to extract insights from review. I am giving you a list of reviews.\
Each element in the list is a review.\
I want you to summarise and return the {no_of_topics} MOST MENTIONED topics (need not separate by sentiment)\
For each key topics, show me 1-2 SHORT PHRASES that you think are most representative of the topic. \
\n
These are the reviews:
{reviews}
\n
ONLY RETURN ME the results in JSON format. DO NOT give me code to do the clustering or your thought process.
\n
Here are the formatting instructions
Make sure the number of topics and the number of lists of sentences ARE THE SAME
'''

class TopicModel(BaseModel):
    topics: List[str] = Field(description="a list of topics")
    sentences: List[List[str]] = Field(description = "a nested list of reviews that are most representative of each topic")

topic_parser = PydanticOutputParser(pydantic_object= TopicModel)
format_instructions = topic_parser.get_format_instructions()

CLUSTER_PROMPT = ChatPromptTemplate(
    messages = [
        HumanMessagePromptTemplate.from_template(template + "\n {format_instructions}")
    ],
    input_variables= ['reviews', 'no_of_topics'],
    partial_variables = {"format_instructions": format_instructions}

)



# Uploading files. 
st.header("Step 1: select a file or upload a file (only works for Microsoft Edge)")
datafile = st.file_uploader("Choose a file")
if datafile is not None:
    file_details = {'FileName': datafile.name, 'FileType': datafile.type}
    if datafile.name.endswith('.csv'):
        df = pd.read_csv(datafile)
    elif datafile.name.endswith('.xlsx'):
        df = pd.read_excel(datafile)
    save_uploaded_file(datafile)
    files.insert(0, datafile.name)
    mapping_file_to_root[datafile.name] = 'data'

st.write('OR')

# Display files in excel or csv format
st.header('Step 1: Select a file')
for root, dirs, filenames in os.walk(data_folder):
    for filename in filenames:
        if filename.endswith(('.csv', '.xlsx')) and filename not in files:
            mapping_file_to_root[filename] = root
            files.append(filename)

selected_file = st.selectbox("Select from available files", files)

if selected_file:
    if 'selected file' in st.session_state:
        if selected_file != st.session_state['selected_file']:
            st.session_state['history'] = []
            st.session_state['generated'] = ['Hello! Ask me anything about ' + selected_file]
            st.session_state['past'] = ['Hey!']
    file_path = os.path.join(mapping_file_to_root[selected_file], selected_file)
    if selected_file.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif selected_file.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    st.write("Data First 5 Rows Preview")
    st.dataframe(df.head(5))

    st.header("Step 2: Select a column for analysis")
    st.write()
    col_name = st.selectbox("Select a column", df.columns)
    service_name = st.radio("Select Analysis", ['Topic Modelling'])

    if service_name == 'Topic Modelling':
        no_of_topics = st.number_input('Enter the number of topics to cluster:', min_value = 1, max_value = 10, value = 6, step = 1)

    service_button = st.button('Run Analysis')

    st.header('Results')

    if service_button:
        result_container = st.empty()
        with st.spinner('Fetching Results'):
            review_list = df[col_name].astype(str).tolist()
            if service_name == 'Topic Modelling':
                llm_chain = LLMChain(llm = OpenAI(model_name = 'gpt-3.5-turbo-16k', openai_api_key = OPEN_API_KEY ), prompt= CLUSTER_PROMPT)
                topics = llm_chain.run(reviews = review_list, no_of_topics = no_of_topics)
                output = topic_parser.parse(topics)
                df = pd.DataFrame({'Topics': output.topics, 'Reviews': output.sentences})
            result_container.write(df)
            # Download button
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine = 'xlsxwriter')
            df.to_excel(writer, index = True)
            writer.save()
            excel_data = output.getvalue()
            b64 = base64.b64encode(excel_data).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download = "result.xlsx">Download Excel</a>'
            st.markdown(href, unsafe_allow_html= True)



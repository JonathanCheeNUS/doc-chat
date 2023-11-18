import base64
from io import BytesIO
import os

import pandas as pd
import streamlit as st

from llm import llm_chain, topic_parser


def save_uploaded_file(uploadedfile):
    with open(os.path.join("data", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to data dir. You can now select the file in Step 1".format(uploadedfile.name))

def xlsx_mime_type(b64_str: str) -> str:
    return f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_str}"

def excel_df_to_b64(df: pd.DataFrame) -> str:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=True)
        writer.save()
        excel_data = output.getvalue()
    return base64.b64encode(excel_data).decode()


def streamlit_dashboard():
    # Variables 
    data_folder = 'data'
    files = []
    mapping_file_to_root = {}

    # Uploading files. 
    st.header("Step 1: select a file or upload a file (only works for Microsoft Edge)")
    datafile = st.file_uploader("Choose a file")
    if datafile is not None:
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
            no_of_topics = st.number_input(
                'Enter the number of topics to cluster:',
                min_value=1,
                max_value=10,
                value=6,
                step=1
            )

        service_button = st.button('Run Analysis')

        st.header('Results')

        if service_button:
            result_container = st.empty()
            with st.spinner('Fetching Results'):
                review_list = df[col_name].astype(str).tolist()
                if service_name == 'Topic Modelling':
                    topics = llm_chain.run(reviews = review_list, no_of_topics = no_of_topics)
                    output = topic_parser.parse(topics)
                    df = pd.DataFrame({'Topics': output.topics, 'Reviews': output.sentences})
                result_container.write(df)

                b64_data = excel_df_to_b64(df)
                href = f'<a href="{xlsx_mime_type(b64_data)}" download="result.xlsx">Download Excel</a>'
                st.markdown(href, unsafe_allow_html= True)

if __name__ == "__main__":
    streamlit_dashboard()

import os
import streamlit as st
import fitz
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import pdfplumber
from PIL import Image
from PIL import Image
from io import BytesIO
import base64
import pandas as pd

load_dotenv()

os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-exp-1206",
    temperature=0,
)

st.set_page_config(layout="wide", page_title="Named Entities Extraction")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, div, span, h1, h2, h3, h4, h5, app-view-root, [class*="css"]  {
        font-family: 'Poppins', sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center;'>Named Entities Extraction</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

def get_entities(formatted_entity_text, text):
    system = """You are a specialization for recognize entities from the given 'Document Context'.  when user give the 'Entity Name' and the 'Additional Instruction Given For Identify Entity', according to the 'Document Context' you have to identify the entity value from the given 'Document Context'. Identify entity value only. nothing else. Give the output as following format for allthe given Entity Names.
    
    Example : 
    
        ***Entity Name 1:*** <Given Entity Name for Entity Name 1>
        ***Additional Instruction Given For Identify Entity 1:*** <Given Additional Instruction for Entity Name 1>
        ***Entity Value 1:*** <Identified Entity Value for Entity Name 1>
        
        ***Entity Name 2:*** <Given Entity Name for Entity Name 2>
        ***Additional Instruction Given For Identify Entity 2:*** <Given Additional Instruction for Entity Name 2>
        ***Entity Value 2:*** <Identified Entity Value for Entity Name 2>

    """
    human = f"""
        {formatted_entity_text}
        Document Context: {text}
    """
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | llm
    text = "What is the capital of France?"
    
    answer = chain.invoke({})
    
    return answer.content

if uploaded_file is not None:
    # Read the PDF file
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()

    # Initialize session state for entities
    if 'entities' not in st.session_state:
        st.session_state.entities = []


    # Input for number of entity extractions
    num_entities = st.number_input('Number of entities to extract', min_value=1, value=1)

    # Determine the number of rows and columns
    num_columns = 2  # Adjust the number of columns as needed
    num_rows = int((num_entities + num_columns - 1) / num_columns)
        
    st.markdown(
        """
        <style>
        [data-testid="stColumn"] {
            padding: 2%;
            box-shadow: rgba(0, 0, 0, 0.05) 0px 0px 0px 1px, rgb(209, 213, 219) 0px 0px 0px 1px inset;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    for row in range(num_rows):
        columns = st.columns(num_columns)
        for i in range(num_columns):
            entity_index = row * num_columns + i
            if entity_index < num_entities:
                with columns[i]:
                    with st.container():
                        st.markdown(f"#### Entity Details {entity_index + 1}")
                        st.text_input("Entity Name", key=f'entity_name_{entity_index}')
                        st.text_area("Additional Context", height=100, key=f'additional_context_{entity_index}')

                
    if st.button('Extract Entities'):
        all_entities_provided = True
        formatted_text = ""

        for entity_index in range(num_entities):
            entity_name = st.session_state.get(f'entity_name_{entity_index}', '')
            additional_context = st.session_state.get(f'additional_context_{entity_index}', '')
            if entity_name:
                formatted_text += f"\n***Entity Name {entity_index + 1}:*** {entity_name} \n***Additional Instruction Given For Identify Entity {entity_index + 1}:*** {additional_context}\n\n\n"
            else:
                st.error(f"Entity Name {entity_index + 1} is required")
                all_entities_provided = False

        if all_entities_provided:
            # Process entity
            entity_value = get_entities(formatted_text, text)
            # st.write(formatted_text)
            # st.write(entity_value)
            
            data = []
            entities = entity_value.split("\n\n")
            for entity in entities:
                if entity.strip():
                    parts = entity.split("\n")
                    entity_name = parts[0].split(":")[1].replace('*', '').strip()
                    additional_context = parts[1].split(":")[1].replace('*', '').strip()
                    entity_value = parts[2].split(":")[1].replace('*', '').strip()
                    data.append([entity_name, additional_context, entity_value])

            df = pd.DataFrame(data, columns=["Entity Name", "Additional Context", "Entity Value"])
            st.table(df)


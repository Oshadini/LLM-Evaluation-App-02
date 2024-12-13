import os
import streamlit as st
import fitz
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-exp-1206",
    temperature=0,
)

st.set_page_config(layout="wide", page_title="Named Entities Extraction")


uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

def get_entities(formatted_entity_text, text):
    system = """You are a specialization for recognizing entities from the given 'Document Context'. When the user provides the 'Entity Name' and the 'Additional Instruction Given For Identify Entity', you have to identify the entity value from the given 'Document Context'. Identify entity value only, nothing else. Provide the output in the following format:

    Example:
        ***Entity Name 1:*** <Given Entity Name for Entity Name 1>
        ***Additional Instruction Given For Identify Entity 1:*** <Given Additional Instruction for Entity Name 1>
        ***Entity Value 1:*** <Identified Entity Value for Entity Name 1>
    """
    human = f"""
        {formatted_entity_text}
        Document Context: {text}
    """
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | llm
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

    # Render rows dynamically
    num_columns = 2
    num_rows = -(-num_entities // num_columns)  # Round up division

    for row in range(num_rows):
        columns = st.columns(num_columns)
        for i in range(num_columns):
            entity_index = row * num_columns + i
            if entity_index < num_entities:
                with columns[i]:
                    st.markdown(f"### Entity Details {entity_index + 1}", unsafe_allow_html=True)
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
            entity_value = get_entities(formatted_text, text)
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

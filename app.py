"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from PIL import Image

import os
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_ORGANIZATION"] = "org-lwaUBVlPJVS50wZwghHFihUA"

from langchain.llms import OpenAI
from ingest_data import embed_doc
from query_data import _template, CONDENSE_QUESTION_PROMPT, QA_PROMPT, get_chain
import pickle
from langchain.callbacks import get_openai_callback
import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
)

def wiki_search(topic):
    page_py = wiki_wiki.page(topic)
    title = page_py.title
    text = page_py.text
    title = title.lower()
    # !!!! IMPORTANT encode and decode to remove non-ascii characters 
    title = title.encode("ascii", "ignore").decode()
    text = text.encode("ascii", "ignore").decode()
    if title not in os.listdir("data"):
        with open(f"data/{title}.txt", "w") as f:
            f.write(text)

# def load_chain():
#     """Logic for loading the chain you want to use should go here."""
#     llm = OpenAI(temperature=0)
#     chain = ConversationChain(llm=llm)
#     return chain

def generate_answer():
    user_input = st.session_state.input
    docs = vectorstore.similarity_search(user_input)
    # if checkbox is checked, print docs

    print(len(docs))
    # PART 2 ADDED: CALLBACK FOR TOKEN USAGE
    with get_openai_callback() as cb:
        output = chain.run(input=user_input, vectorstore = vectorstore, context=docs[:2], chat_history = [], question= user_input, QA_PROMPT=QA_PROMPT, CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT, template=_template)
        print(cb.total_tokens)
    

    st.session_state.past.append(user_input)
    # print(st.session_state.past)
    st.session_state.generated.append(output)
    
    print(st.session_state.generated)
    # PART2 ADDED
    # if st.session_state.generation includes "related topics:" remove that from st.session_state.generation and add it to a new list
    if "#" in st.session_state.generated[-1]:
        st.session_state.generated[-1], st.session_state.topics = st.session_state.generated[-1].split("#")[0], st.session_state.generated[-1].split("#")[1]
        
    print(st.session_state.generated)
    print(st.session_state.topics)
    print(type(st.session_state.topics))

    with open("topics.txt", "w") as f:
        for char in st.session_state.topics:
            if char == "[" or char == "]" or char == "'":
                continue
            else:
                f.write(char)


def rebuild_index():
    with st.spinner('Cramming documents... Hold on! This may take a while...'):
        embed_doc()
        with open("vectorstore.pkl", "rb") as f:
            vectorstore = pickle.load(f)
            print("Loading vectorstore...")
        chain = get_chain(vectorstore)



# From here down is all the StreamLit UI.
im = Image.open('content/App_Icon.png')
st.set_page_config(page_title="NakheelGPT", page_icon=im)

# hide_default_format = """
#        <style>
#        #MainMenu {visibility: hidden; }
#        footer {visibility: hidden;}
#        </style>
#        """
# st.markdown(hide_default_format, unsafe_allow_html=True)

st.sidebar.title(":darkblue[NakheelGPT Demo]")
st.sidebar.caption("Next-Gen ChatBot built on top of the state of the art AI model - ChatGPT.")

with st.sidebar.expander("Upload a document you would like to chat about! 🚀"):
    uploaded_file = st.file_uploader("Upload",type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="hidden")

    # check if file is uploaded and file does not exist in data folder
    if uploaded_file is not None and uploaded_file.name not in os.listdir("data"):
        # write the file to data directory
        with open("data/" + uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write("File uploaded successfully")
        with st.spinner('Cramming document...'):
            embed_doc()
    # open vectorstore.pkl if it exists in current directory
    if "vectorstore.pkl" in os.listdir("."):
        with open("vectorstore.pkl", "rb") as f:
            
            vectorstore = pickle.load(f)
            print("Loading vectorstore...")

        chain = get_chain(vectorstore)

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []



# placeholder = st.empty()

st.text_input("Talk to NakheelGPT: ", value="",  key="input", on_change=generate_answer)

st.sidebar.markdown("***")
st.sidebar.markdown("Do these topics interest you? Click the button below to add it's wiki articles to my knowledge base 🧠")
# st.markdown('###')


# PART 2 ADDED: BUTTONS FOR WIKI ARTICLES
# buttons need to be in a separate column
col1, col2, col3 = st.sidebar.columns(3)
if "topics.txt" in os.listdir("."):
    with open("topics.txt", "r") as f:
        topics = f.read().split(",")
        print(topics)
        if col1.button(topics[0]):
            wiki_search(topics[0])
            rebuild_index()
        if col2.button(topics[1]):
            wiki_search(topics[1])
            rebuild_index()
        if col3.button(topics[2]):
            wiki_search(topics[2])
            rebuild_index()

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        
        message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed="Work")
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style="initials", seed="RD")
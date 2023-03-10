"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
import yaml
from yaml.loader import SafeLoader
from PIL import Image

import os
from ingest_data import embed_doc
from query_data import _template, CONDENSE_QUESTION_PROMPT, QA_PROMPT, get_chain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
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


def generate_answer():
    user_input = st.session_state.input
    docs = vectorstore.similarity_search(user_input, k=20)

    print(len(docs))
    # PART 2 ADDED: CALLBACK FOR TOKEN USAGE
    with get_openai_callback() as cb:
        output = chain.run(input=user_input, vectorstore = vectorstore, context=docs, chat_history = [], question= user_input, QA_PROMPT=QA_PROMPT, CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT, template=_template)
        print(cb.total_tokens)
    

    st.session_state.past.append(user_input)
    # print(st.session_state.past)
    st.session_state.generated.append(output)
    ##ADDED FOR TESTING
    if "#" in st.session_state.generated[-1]:
        st.session_state.generated[-1], st.session_state.topics = st.session_state.generated[-1].split("#")[0], st.session_state.generated[-1].split("#")[1]
    
    with open("topics.txt", "w") as f:
        for char in st.session_state.topics:
            if char == "[" or char == "]" or char == "'":
                continue
            else:
                f.write(char)
    print(st.session_state.generated)


def rebuild_index():
    with st.spinner('Cramming documents... Hold on! This may take a while...'):
        embed_doc()
        print("Loading vectorstore...")
        chain = get_chain(vectorstore)


def encrypt_password():
    with open("config.yaml") as file:
        config = yaml.load(file, Loader=SafeLoader)
    #     for user in config["credentials"]["usernames"]:
    #         key = config["credentials"]["usernames"][user]["password"]
    #         if key.startswith("$") == False:
    #             enc_key = stauth.Hasher([key]).generate()
    #             print(enc_key[0])
    #             config["credentials"]["usernames"][user]["password"] = enc_key[0]

    # with open("config.yaml", "w") as file:
    #     file.write(yaml.dump(config, default_flow_style=False))

    return config

# From here down is all the StreamLit UI.
im_icon = Image.open('content/nakheel_icon.png')
st.set_page_config(page_title="NakheelGPT", page_icon=im_icon)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="db/", embedding_function=embeddings)        
print("Loaded vectorstore...")
chain = get_chain(vectorstore)

# Authentication Setup
config = encrypt_password()

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config["preauthorized"],
)

name, authentication_status, username = authenticator.login("Login", "main")


hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """

if authentication_status:
    st.markdown(hide_default_format, unsafe_allow_html=True)
    st.title("NakheelGPT")
    st.caption("Next-Gen ChatBot built on top of the state of the art AI model - ChatGPT.")
    st.markdown("#")
    initials = ''.join([x[0].upper() for x in st.session_state["name"].split(' ')])
    
    st.sidebar.write(f'# Welcome {st.session_state["name"]}')
    authenticator.logout("Logout", "sidebar")
    st.sidebar.markdown("###")
    with st.sidebar:
        selected = option_menu(None, ["NakheelGPT","Upload", 'Settings'], 
                               icons=['chat-dots-fill','cloud-arrow-up-fill', 'gear-fill'], menu_icon="cast", default_index=0)
    
    
    if selected == "Upload":
        uploaded_file = st.file_uploader("Upload a document you would like to chat about! 🚀",type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        # check if file is uploaded and file does not exist in data folder
        if uploaded_file is not None and uploaded_file.name not in os.listdir("data"):
            # write the file to data directory
            with open("data/" + uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.write("File uploaded successfully")
            with st.spinner('Cramming documents into silicon...'):
                embed_doc()
    elif selected == "Settings":
        st.markdown("### Still building this page, come back later... 🛠️")
    else:
        if "generated" not in st.session_state:
            st.session_state["generated"] = []

        if "past" not in st.session_state:
            st.session_state["past"] = []

        st.markdown("Are you interested in these topics? Click to add their wiki articles to my knowledge base  🧠")

        # PART 2 ADDED: BUTTONS FOR WIKI ARTICLES
        # buttons need to be in a separate column
        col1, col2, col3 = st.columns(3)
        if "topics.txt" in os.listdir("."):
            with open("topics.txt", "r") as f:
                topics = f.read().split(",")
                if len(topics) >= 3: 
                    print(topics)
                    if col1.button(topics[0],use_container_width=True):
                        wiki_search(topics[0])
                        rebuild_index()
                    if col2.button(topics[1],use_container_width=True):
                        wiki_search(topics[1])
                        rebuild_index()
                    if col3.button(topics[2],use_container_width=True):
                        wiki_search(topics[2])
                        rebuild_index()

        st.markdown("#")          
        st.text_input("**Chat with NakheelGPT:**", value="",  key="input", on_change=generate_answer)

        if st.session_state["generated"]:

            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed="nakheel")
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style="initials", seed=initials)
    
    st.sidebar.markdown("***")
    im_logo = Image.open("content/nakheel_logo.png")
    st.sidebar.image(im_logo, use_column_width='auto')
elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')
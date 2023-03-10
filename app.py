"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from PIL import Image

import os
from langchain.llms import OpenAI
from ingest_data import embed_doc
from query_data import _template, CONDENSE_QUESTION_PROMPT, QA_PROMPT, get_chain
import pickle
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
import wikipediaapi


wiki_doc = wikipediaapi.Wikipedia(
    language="en", extract_format=wikipediaapi.ExtractFormat.WIKI
)


# def register_user():
#     if st.button('New User? Register here'):
#         try:
#             if authenticator.register_user('Register user', preauthorization=False):
#                 st.success('User registered successfully')
#         except Exception as e:
#             st.error(e)


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


def wiki_search(topic):
    page_py = wiki_doc.page(topic)
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
        output = chain.run(
            input=user_input,
            vectorstore=vectorstore,
            context=docs,
            chat_history=[],
            question=user_input,
            QA_PROMPT=QA_PROMPT,
            CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT,
            template=_template,
        )
        print(cb.total_tokens)

    st.session_state.past.append(user_input)
    # print(st.session_state.past)
    st.session_state.generated.append(output)
    ##ADDED FOR TESTING
    if "#" in st.session_state.generated[-1]:
        st.session_state.generated[-1], st.session_state.topics = (
            st.session_state.generated[-1].split("#")[0],
            st.session_state.generated[-1].split("#")[1],
        )

    with open("topics.txt", "w") as f:
        for char in st.session_state.topics:
            if char == "[" or char == "]" or char == "'":
                continue
            else:
                f.write(char)
    print(st.session_state.generated)


def rebuild_index():
    with st.spinner("Cramming documents... Hold on! This may take a while..."):
        embed_doc()
        with open("vectorstore.pkl", "rb") as f:
            vectorstore = pickle.load(f)
            print("Loading vectorstore...")
        chain = get_chain(vectorstore)




print("0**************************************")

# From here down is all the StreamLit UI.
im_icon = Image.open("content/nakheel_icon.png")
st.set_page_config(page_title="NakheelGPT", page_icon=im_icon)

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
print("1**************************************")


####
# Authentication Setup
config = encrypt_password()
print("2**************************************")

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config["preauthorized"],
)

name, authentication_status, username = authenticator.login("Login", "main")
print("3**************************************")


if authentication_status:
    vectorstore = Chroma(persist_directory="db/", embedding_function=OpenAIEmbeddings())
    print("Loaded vectorstore...")
    chain = get_chain(vectorstore)
    
    print("4################################################")
    st.title("NakheelGPT")
    st.caption(
        "Next-Gen ChatBot built on top of the state of the art AI model - ChatGPT."
    )
    st.markdown("###")

    st.sidebar.write(f'# Welcome {st.session_state["name"]}!')
    # initials = ''.join([x[0].upper() for x in st.session_state["name"].split(' ')])
    initials = 'YM'

    authenticator.logout("Logout", "sidebar")
    st.sidebar.markdown("***")    
    st.sidebar.markdown("###")    
    print("4.5################################################")

    with st.sidebar.expander("Upload a document you would like to chat about 🚀"):
        uploaded_file = st.file_uploader(
            "Upload",
            type=None,
            accept_multiple_files=False,
            key=None,
            help=None,
            on_change=None,
            args=None,
            kwargs=None,
            disabled=False,
            label_visibility="hidden",
        )
        print("4.75################################################")
        # check if file is uploaded and file does not exist in data folder
        if uploaded_file is not None and uploaded_file.name not in os.listdir("data"):
            # write the file to data directory
            with open("data/" + uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.write("File uploaded successfully")
            with st.spinner("Cramming document..."):
                embed_doc()
    print("5################################################")
    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    st.markdown(
        "Do these topics interest you? Click the button below to add it's wiki articles to my knowledge base 🧠"
    )

    # PART 2 ADDED: BUTTONS FOR WIKI ARTICLES
    # buttons need to be in a separate column
    col1, col2, col3 = st.columns(3)
    if "topics.txt" in os.listdir("."):
        with open("topics.txt", "r") as f:
            topics = f.read().split(",")
            if len(topics) >= 3:
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
    print("6################################################")
    st.markdown("#")
    st.text_input("Talk to NakheelGPT: ", value="", key="input", on_change=generate_answer)

    st.sidebar.markdown("#")

    im_logo = Image.open("content/nakheel_logo.png")
    st.sidebar.image(im_logo, use_column_width="auto")

    if st.session_state["generated"]:
        print("7################################################")
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):

            message(
                st.session_state["generated"][i],
                key=str(i),
                avatar_style="bottts",
                seed="Work",
            )
            message(
                st.session_state["past"][i],
                is_user=True,
                key=str(i) + "_user",
                avatar_style="initials",
                seed=initials,
            )

elif authentication_status == False:
    st.error("Username/password is incorrect")
    # register_user()


elif authentication_status == None:
    st.warning("Please enter your username and password")
    # register_user() 

from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI, OpenAIChat
from langchain.chains import ChatVectorDBChain

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.


Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
prefix_messages = [{"role": "system", "content": "You are 'NakheelGPT', an AI assistant that answers questions in a concise manner."}]

template = """
You are given the following extracted parts of a long document and a question.
At the end of your answer, add a newline and return a python list of up to three wikipedia topics which are related to the context and question leading with a "#" like this wihout mentioning anything else:
#['topic1', 'topic2', 'topic3']

If you don't know the answer, don't try to make up an answer.


Question: {question}
=========
{context}
=========
Answer in Markdown:"""

QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    llm = OpenAIChat(temperature=0, prefix_messages=prefix_messages)
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT
    )
    return qa_chain

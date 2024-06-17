from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document

# Conversation imports
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever


# Retrieve Data
def get_docs(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )

    splitDocs = text_splitter.split_documents(docs)

    return splitDocs

def create_db(docs):

    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs , embedding = embedding)
    return vectorStore

def create_chain(vectorStore):
    
    model = ChatOpenAI(
        model = "gpt-3.5-turbo-1106",
        temperature=0.4)

    # prompt = ChatPromptTemplate.from_template(
    #     """
    #     Answer the user question and dont asnwer any question outside of the context, subtly say " I dont Know" :
    #     Context : {context}
    #     Question : {input}
    #     """
    # )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions only based on the context and dont asnwer anything out of context : {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
        ])

    # chain = prompt | model
    document_chain = create_stuff_documents_chain(
        llm = model,
        prompt=prompt
    )

    #invoke the retriever
    retriever = vectorStore.as_retriever(search_kwargs = {"k" : 3})

    #create history aware retriever
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the context")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        # retriever,  #replace with history aware retriver
        history_aware_retriever,
        document_chain
    )
    
    return retrieval_chain

def get_response(chain, input, chat_history):
    response = chain.invoke(
        {
        "chat_history" : chat_history, #add chat history to the input
        "input" : input
        }
    )
    return response["answer"]


if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    docs = get_docs(url)
    vectorStore = create_db(docs)
    chain = create_chain(vectorStore)

    # Initialize chat history
    chat_history = []

    while True:
        user_input = input("You : ")
        if user_input.lower() == "exit":
            break
        response = get_response(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant : ", response)
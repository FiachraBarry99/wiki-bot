import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

def answer_prompt(prompt, session_id):
    # initialise llm
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # create retriever
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(
            index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    retriever = vectorstore.as_retriever()

    # allow retriever to be aware of history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # answer question
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you cannot answer the question based on the given "
        "context, say that you don't know and the question is not relevant."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # manage chat history
    store = {}      # simplify this so history is passed in as a dict or list

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    res = conversational_rag_chain.invoke(
        {"input": prompt},
        config={
            "configurable": {"session_id": session_id}
        },
    )

    return res

    

# example invokation
if __name__ == "__main__":
    INPUT = "How many islands are in the Aran Islands?"
    res = answer_prompt(prompt=INPUT, session_id='abc123')

    print(res["answer"])
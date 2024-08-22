import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage

load_dotenv()

def answer_prompt(prompt: str, chat_history: list[list[str, str]] = []):
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

    res = rag_chain.invoke({"input": prompt,"chat_history": chat_history})
    
    return res

    

# example invokations
if __name__ == "__main__":

    # initialise chat history
    chat_hist = []

    # invoke chain
    INPUT1 = "How many people live on Inis Mor?"
    res = answer_prompt(prompt=INPUT1, chat_history=chat_hist)

    print(res["answer"])

    # add prompt and result to chat history
    new_hist = [HumanMessage(content=INPUT1), res["answer"]]
    chat_hist.extend(new_hist)


    # invoke chain again with history specific question
    INPUT2 = "How big is it?"
    res = answer_prompt(prompt=INPUT2, chat_history=chat_hist)

    print(res["answer"])

    new_hist = [HumanMessage(content=INPUT2), res["answer"]]
    chat_hist.extend(new_hist)
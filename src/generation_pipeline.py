# Generation pipeline: the final stage of RAG (ingest → retrieve → generate).
# Includes multi-turn memory — we manually maintain a chat_history list and
# pass it into the prompt each turn so Claude sees the full conversation.

from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
# pyrefly: ignore [missing-import]
from config import AGENT_MODEL
from retrieval_pipeline import load_vectorstore, get_retriever

load_dotenv()


def format_docs(docs: list) -> str:
    # Joins all retrieved chunks into a single string block to pass as context.
    # Each chunk is separated by a blank line so the LLM can clearly distinguish between them.
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, model_name: str = AGENT_MODEL):
    # MessagesPlaceholder inserts the chat_history list into the prompt between
    # the system message and the current human question.
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer using only the context below.
Reply naturally and directly — do not mention the context, do not say "based on the provided context" or similar phrases.
If the answer is not in the context, just say "I don't know."

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    llm = ChatAnthropic(model=model_name)

    # Input is a dict with three keys: question, chat_history, and context.
    # We extract ["question"] with a lambda before passing it to the retriever.
    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def main():
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)
    chain = build_rag_chain(retriever)

    # chat_history holds the full conversation as alternating HumanMessage/AIMessage objects.
    # We pass it into every chain.invoke() so Claude can refer back to earlier exchanges.
    chat_history = []

    print("\nRAG Chatbot ready. Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        answer = chain.invoke({"question": question, "chat_history": chat_history})
        print(f"Agent: {answer}\n")

        # Append this turn to history so the next question sees it
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))


if __name__ == "__main__":
    main()

import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from vecotr_store import retriever  # Ensure this works and returns a retriever
from promts import prompt
from dotenv import load_dotenv
import os 
load_dotenv()


# # Define the system prompt
# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise.\n\n"
#     "{context}"
# )

# # Create the prompt template
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# Initialize the LLM
llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=os.getenv("GROG_API_KEY")
)

# Create the chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Streamlit UI
st.set_page_config(page_title="RAG Q&A App", layout="centered")

st.title("🔍 Question Answering Assistant")
st.write("Ask any question based on your knowledge base.")

# User input
user_input = st.text_input("Enter your question:", "")

if st.button("Get Answer"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({"input": user_input})
                st.success("Answer:")
                st.write(response.get("answer", "No answer returned."))
            except Exception as e:
                st.error(f"An error occurred: {e}")



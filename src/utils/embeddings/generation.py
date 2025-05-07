from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from vecotr_store import retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os 
from dotenv import load_dotenv

load_dotenv()

from rag_advanced.src.utils.prompts.promts import prompt


llm=ChatGroq(model="llama3-70b-8192", api_key=os.getenv("GROG_API_KEY"))


question_answer_chain = create_stuff_documents_chain(llm, prompt )
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "how can we open doors from interior"})
print(response["answer"])
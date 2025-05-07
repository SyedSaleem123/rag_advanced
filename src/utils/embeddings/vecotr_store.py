from langchain_community.vectorstores import FAISS
from chunks import text_chunks 
from embedding import embeddings

vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

# Step 5: Save the FAISS index to disk
vector_store.save_local("faiss_index_store")

print(vector_store)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})

retrieved_docs = retriever.invoke("how can we open doors from interior  ")
print(retrieved_docs)
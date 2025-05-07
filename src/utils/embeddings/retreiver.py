from vecotr_store import vector_store
def retriever1(search_type="similarity", search_kwargs={"k":3}):
    retriever = vector_store.as_retriever(search_type=search_type, search_kwargs= search_kwargs)
    return retriever

retrieved_docs = retriever1.invoke("how can we open doors from interior  ")
# print(retrieved_docs)
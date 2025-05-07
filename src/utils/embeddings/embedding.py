from langchain.embeddings import HuggingFaceEmbeddings

#Download the Embeddings from Hugging Face
def download_hugging_face_embeddings():
    embeddinds=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddinds


embeddings=download_hugging_face_embeddings()

# query=embeddings.embed_query("helloworld")
# print("Embedding vector dimension:", len(query))

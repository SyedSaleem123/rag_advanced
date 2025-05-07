from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Extract Data From the url File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents


extracted_data=load_pdf_file(data=r"C:\Users\AI_ML PC_4\Desktop\Tesla_rag\resources\data")

print(extracted_data[0])

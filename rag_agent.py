from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from langchain_core.tools import tool
from langchain.agents import create_agent


model = ChatOllama(model='llama3.2:1b',
                   base_url=os.getenv(
                   "OLLAMA_BASE_URL",
                   "http://localhost:11434"
                    ),
                   temperature= 0.2)


loader = DirectoryLoader(
    path = 'books',
    glob = '*.pdf',
    loader_cls=PyPDFLoader
)

doc = loader.load()
print(len(doc))

split = RecursiveCharacterTextSplitter(
    chunk_size = 600,
    chunk_overlap = 100
)

chunks = split.split_documents(doc)
print(len(chunks))

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

def clean(text: str) -> str:
    return text.encode("utf-8", "ignore").decode("utf-8")

cleaned_chunks= []
for c in chunks:
    c.page_content = c.page_content.encode("utf-8", "ignore").decode("utf-8").replace("\x00", " ").strip()
    if len(c.page_content) > 20:
        cleaned_chunks.append(c)


VB = FAISS.from_documents(
    cleaned_chunks,
    embedding
)

retrieve = VB.as_retriever(search_type = 'similarity', search_kwargs = {'k':2})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@tool
def search_query(query: str) -> str:
    """Give relevant answer"""
    document = retrieve.invoke(query)
    context = format_docs(document)
    return context


agent = create_agent(
    model = model,
    tools = [search_query],
    system_prompt=(
    """You are a AI expert advisor please answer carefully using uploaded query"""
    )
)

res = agent.invoke({
    "messages":[
        {
            'role':"user",
            "content":"What is an AI Agent? "
        }
    ]
})

#print(res["messages"][-1].content)
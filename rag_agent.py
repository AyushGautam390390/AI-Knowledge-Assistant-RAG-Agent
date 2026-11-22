from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from langchain_core.tools import tool
from langchain.agents import create_agent
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithoutReference
    
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import pandas as pd
from ragas.run_config import RunConfig
from sentence_transformers import CrossEncoder


model = ChatOllama(model="llama3.2:3b",
                   base_url=os.getenv(
                   "OLLAMA_BASE_URL",
                   "http://localhost:11434"
                    ),
                   temperature= 0.2)
judge_model = ChatOllama(model="qwen2.5:7b", temperature=0.2, format='json')

loader = DirectoryLoader(
    path = 'books',
    glob = '*.pdf',
    loader_cls=PyPDFLoader
)

doc = loader.load()
print(len(doc))

split = RecursiveCharacterTextSplitter(
    chunk_size = 900,
    chunk_overlap = 120
)

chunks = split.split_documents(doc)
print(len(chunks))

embedding = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5",
    model_kwargs = {"device":"cpu"},
    encode_kwargs = {'normalize_embeddings':True}
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

retrieve = VB.as_retriever(search_type = 'mmr', search_kwargs = {'k':5})

reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")

# fetch more candidates, then rerank down to 5
base_retriever = VB.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 20, "fetch_k": 50}
)

def retrieve_rerank(query: str, top_n: int = 5):
    docs = base_retriever.invoke(query)
    scores = reranker.predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@tool
def search_query(query: str) -> str:
    """Give relevant answer"""
    
    return format_docs(retrieve_rerank(query))


agent = create_agent(
    model = model,
    tools = [search_query],
    system_prompt=(
    "You are an AI expert advisor. ALWAYS call the search_query tool first, "
            "then answer ONLY from the retrieved context. "
            "If the context does not contain the answer, say you don't know."
        ),
    
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

# Test questions
test_questions = [
    "What is an AI Agent?",
    "What is Retrieval Augmented Generation?",
    "What is a vector database?"
]

questions = []
answers = []
contexts = []

for question in test_questions:

    # Retrieve relevant document chunks
    docs = retrieve_rerank(question)
    context = [doc.page_content for doc in docs]

    # Generate the answer using your existing agent
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    answer = result["messages"][-1].content

    questions.append(question)
    answers.append(answer)
    contexts.append(context)


# Create RAGAS evaluation dataset
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts
})


# Configure the evaluator using your existing Ollama model
judge_llm = LangchainLLMWrapper(judge_model)
judge_embeddings = LangchainEmbeddingsWrapper(embedding)


# Evaluate RAG performance
metrics = [
    Faithfulness(),
    AnswerRelevancy(),
    LLMContextPrecisionWithoutReference(),
    
]


run_config = RunConfig(
    timeout=600,
    max_retries=2,
    max_wait=30,
    max_workers=1,
    log_tenacity=True
)

results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=judge_llm,
    embeddings=judge_embeddings,
    run_config=run_config,
    raise_exceptions=True
)

# Print individual results
print("\n===== RAGAS RESULTS =====")
print(results.to_pandas().to_string(index=False))

# Print average metric scores
df = results.to_pandas()

metric_names = [
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_without_reference"
]

print("\n===== AVERAGE SCORES =====")
print(df[metric_names].mean().round(3))

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_community.llms import Replicate
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from deep_translator import GoogleTranslator

# Inicializar FastAPI
app = FastAPI()

# Configurar CORS para que el frontend pueda acceder a la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las conexiones (puedes restringirlo)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir el modelo de entrada para la API
class QuestionRequest(BaseModel):
    question: str

# Definir el modelo de lenguaje
llm = Replicate(
    model="meta/llama-2-13b-chat",
    model_kwargs={
        "temperature": 0.75,
        "max_length": 500,
        "top_p": 1,
        "language": "es"
    },
)

# Definir el prompt para la cadena de preguntas y respuestas
template = """Usa el siguiente contexto para responder siempre en español.
Se breve y conciso, máximo tres oraciones y responde solamente a la pregunta, no des info adicional.
Es la página de un profesor. Habla en tercera persona.

Si la info es incorrecta o no está clara, menciona que no tienes suficiente info.

{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Función para obtener los documentos de la web y PDFs
def load_documents():
    # Web Scraping
    url = "https://cs.uns.edu.ar/~dcm/home/"
    def custom_extractor(html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text()
    
    loader = RecursiveUrlLoader(url=url, extractor=custom_extractor)
    web_docs = loader.load()

    # Descargar PDFs
    pdf_urls = [
        "https://cs.uns.edu.ar/~dcm/downloads/Pautas%20Examen%20Final%20Cursado%202017.pdf",
        "https://cs.uns.edu.ar/~dcm/downloads/Pautas%20Examen%20Final%20Cursado%202016.pdf",
        "https://cs.uns.edu.ar/~dcm/downloads/PautasExamenLibreTdP.pdf"
    ]
    pdf_docs = []

    for pdf_url in pdf_urls:
        pdf_filename = pdf_url.split("/")[-1]
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(pdf_filename, "wb") as f:
                f.write(response.content)
            pdf_loader = PyPDFLoader(pdf_filename)
            pdf_docs.extend(pdf_loader.load())

    return web_docs + pdf_docs

# Cargar documentos y configurar FAISS
docs = load_documents()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=115, chunk_overlap=10)
all_splits = text_splitter.split_documents(docs)

hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
vectorstore = FAISS.from_documents(documents=docs, embedding=hf)
retriever = VectorStoreRetriever(vectorstore=vectorstore, search_kwargs={"k": 3})

# Crear la cadena de preguntas y respuestas
qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=retriever, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# Endpoint para responder preguntas
@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question

    # Traducir la pregunta al inglés
    translated_query = GoogleTranslator(source="auto", target="en").translate(question)

    # Obtener respuesta del modelo
    response = qa_chain.invoke(translated_query)

    # Traducir la respuesta de vuelta al español
    translated_response = GoogleTranslator(source="auto", target="es").translate(response["result"])

    return {"answer": translated_response}

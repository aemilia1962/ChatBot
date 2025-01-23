import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

# ฟังก์ชันสำหรับโหลดไฟล์ทั้งหมดในโฟลเดอร์
def load_documents_from_folder(folder_path: str):
    """
    โหลดไฟล์ทั้งหมดจากโฟลเดอร์ที่ระบุ (รองรับไฟล์ .txt และ .md)
    Args:
        folder_path (str): พาธไปยังโฟลเดอร์
    Returns:
        list: รายการเอกสารที่โหลดทั้งหมด
    """
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and (filename.endswith('.txt') or filename.endswith('.md')):  # รองรับ .txt และ .md
            loader = TextLoader(file_path)
            documents.extend(loader.load())  # โหลดเอกสารจากไฟล์นี้
    return documents


# Initial Vector Store setup
folder_path = 'data'  # ระบุโฟลเดอร์ที่ต้องการโหลดไฟล์
documents = load_documents_from_folder(folder_path)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents)

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=split_docs, embedding=embedding_model)

llm = ChatOllama(model="llama3.2")

prompt_template = ChatPromptTemplate.from_template(
    "You are a knowledgeable and friendly assistant for TechBerry. "
    "Your job is to provide clear, helpful, and natural-sounding answers based on the provided context. "
    "Even if the question isn't perfectly clear, try your best to infer what the user is asking and provide a relevant response. "
    "If you cannot find relevant information in the context, politely let the user know and encourage them to ask another question. "
    "Never generate information outside the provided context. "
    "Context: {context}\nQuestion: {question}\nAnswer:"
)

def update_knowledge_base(file_path: str):
    """
    อัปเดต Knowledge Base ด้วยไฟล์ใหม่
    Args:
        file_path (str): พาธไฟล์ที่อัปโหลด
    """
    if not os.path.abspath(file_path).startswith(os.path.abspath('data/')):
        raise ValueError("Files must be located in the 'data/' directory.")
    
    loader = TextLoader(file_path)
    new_documents = loader.load()
    
    split_docs = text_splitter.split_documents(new_documents)
    global vectorstore
    vectorstore.add_documents(split_docs)

def rag_chain(question: str) -> str:
    related_docs = vectorstore.similarity_search(question)
    if not related_docs:
        return "I'm sorry, I couldn't find any relevant information in the provided context."

    context = "\n".join(doc.page_content for doc in related_docs)
    prompt = prompt_template.format(context=context, question=question)
    response = llm.invoke(prompt)
    return response.content.strip() if response.content.strip() else "I'm sorry, I couldn't generate a response based on the context."

def update_instruction(new_instruction: str):
    """
    Update the global instruction prompt template.
    Args:
        new_instruction (str): The new instruction text.
    """
    global prompt_template
    prompt_template = ChatPromptTemplate.from_template(
        f"{new_instruction}\nContext: {{context}}\nQuestion: {{question}}\nAnswer:"
    )

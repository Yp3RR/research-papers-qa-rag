import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

st.title("📚 Research Paper Q&A System")
st.markdown("Upload research papers and ask questions!")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return pipeline("text-generation", model="google/flan-t5-base", max_length=512, device=-1)

embeddings = load_embeddings()
llm = load_llm()

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    st.success(f"✅ PDF processed! {len(chunks)} chunks created.")
    
    question = st.text_input("Ask a question about the paper:")
    
    if question:
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = f"Answer based on context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        answer = llm(prompt, max_length=200)[0]['generated_text']
        
        st.subheader("Answer:")
        st.write(answer)
        
        st.subheader("Sources:")
        for i, doc in enumerate(docs):
            with st.expander(f"Chunk {i+1}"):
                st.write(doc.page_content)
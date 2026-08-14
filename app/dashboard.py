import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

st.title(" Research Paper Q&A System")
st.markdown("Upload a research paper and ask questions about it!")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        st.error("⚠️ GROQ_API_KEY is missing! Add it under Streamlit Cloud -> Settings -> Secrets.")
        st.stop()

    return Groq(api_key=api_key)

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

        prompt = f"""You are a research assistant. Answer the question using ONLY the context provided below. If the answer isn't in the context, say "I couldn't find this in the document."

Context:
{context}

Question: {question}

Answer:"""

        response = llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content

        st.subheader("Answer:")
        st.write(answer)

        st.subheader("Sources:")
        for i, doc in enumerate(docs):
            with st.expander(f"Chunk {i+1}"):
                st.write(doc.page_content)
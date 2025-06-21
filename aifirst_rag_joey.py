import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import pandas as pd
import docx
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import uuid

# --- PAGE CONFIG ---
st.set_page_config(page_title="AIFirst RAG Assistant", page_icon="📚", layout="wide")
st.title("🔍 AIFirst RAG Assistant")
st.markdown("Upload PDFs, DOCX, Excel, TXT, and more — then ask natural language questions.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("🔐 Configuration")

# Input fields
openai_key_input = st.sidebar.text_input("OpenAI API Key", type="password")
qdrant_api_input = st.sidebar.text_input("Qdrant API Key", type="password")
qdrant_url_input = st.sidebar.text_input(
    "Qdrant URL (e.g., https://yourhost.cloud:6333)",
    value="https://6a7820c2-43e6-45f7-bd2e-6e1f73bc6906.eu-central-1-0.aws.cloud.qdrant.io:6333"
)

# Session state defaults
for key in ["openai_valid", "qdrant_valid", "qdrant_client", "openai_client"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Validate on button click
if st.sidebar.button("🔄 Connect & Validate"):
    # OpenAI validation
    try:
        openai_client = OpenAI(api_key=openai_key_input)
        openai_client.models.list()
        st.session_state["openai_valid"] = True
        st.session_state["openai_client"] = openai_client
    except Exception as e:
        st.session_state["openai_valid"] = False
        st.sidebar.error(f"❌ OpenAI key error: {e}")

    # Qdrant validation
    try:
        qdrant_client = QdrantClient(url=qdrant_url_input, api_key=qdrant_api_input)
        qdrant_client.get_collections()
        st.session_state["qdrant_valid"] = True
        st.session_state["qdrant_client"] = qdrant_client
    except Exception as e:
        st.session_state["qdrant_valid"] = False
        st.sidebar.error(f"❌ Qdrant error: {e}")

# Show validation results
if st.session_state["openai_valid"] is True:
    st.sidebar.success("✅ OpenAI API key is valid!")
elif st.session_state["openai_valid"] is False:
    st.sidebar.error("❌ Invalid OpenAI API key")

if st.session_state["qdrant_valid"] is True:
    st.sidebar.success("✅ Qdrant connected!")
elif st.session_state["qdrant_valid"] is False:
    st.sidebar.error("❌ Qdrant not connected")

# --- HALT IF NOT VALIDATED ---
if not st.session_state.get("openai_valid"):
    st.warning("Please validate your OpenAI API key to continue.")
    st.stop()

if not st.session_state.get("qdrant_valid"):
    st.warning("Please validate your Qdrant credentials to continue.")
    st.stop()

# --- INITIALIZE CLIENTS AND SETTINGS ---
client = st.session_state["openai_client"]
qdrant = st.session_state["qdrant_client"]
COLLECTION_NAME = "rag_demo"

# --- INIT COLLECTION IF NEEDED ---
try:
    existing_collections = qdrant.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in existing_collections):
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
except Exception as e:
    st.error(f"Failed to initialize Qdrant collection: {e}")
    st.stop()


# --- EMBEDDING MODEL ---
@st.cache_resource
def get_embedder():
    """Load the sentence transformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = get_embedder()

# --- FILE EXTRACTOR ---
def extract_text_from_file(uploaded_file):
    """Extracts text from various file formats."""
    file_type = uploaded_file.name.split(".")[-1].lower()
    try:
        if file_type == "txt":
            return uploaded_file.read().decode("utf-8")

        elif file_type == "pdf":
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                return "\n\n".join([page.get_text() for page in doc])

        elif file_type == "docx":
            doc = docx.Document(uploaded_file)
            return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])

        elif file_type in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
            return df.to_string(index=False)

        elif file_type == "csv":
            df = pd.read_csv(uploaded_file)
            return df.to_string(index=False)

        elif file_type == "html":
            soup = BeautifulSoup(uploaded_file.read(), "html.parser")
            return soup.get_text()

        else:
            st.warning(f"Unsupported file type: {file_type}")
            return ""
    except Exception as e:
        st.error(f"❌ Failed to extract text from {uploaded_file.name}: {e}")
        return ""

# --- UI FOR UPLOAD AND PROCESSING ---
st.header("1. Upload & Process Document")

# ADDED: A number input to control the minimum chunk size
min_chunk_size = st.number_input(
    "Minimum Chunk Size (characters)",
    min_value=20,
    max_value=1000,
    value=50,  # Default value
    step=10,
    help="Chunks of text smaller than this size will be discarded to filter out noise."
)

uploaded_file = st.file_uploader(
    "Upload a document (PDF, DOCX, Excel, CSV, TXT, HTML)", 
    type=["txt", "pdf", "docx", "xlsx", "xls", "csv", "html"]
)

if uploaded_file:
    text_content = extract_text_from_file(uploaded_file)

    if not text_content:
        st.warning("⚠️ No extractable text found in the uploaded document.")
    else:
        # UPDATED: The chunking logic now uses the value from the number input
        chunks = [
            chunk.strip() 
            for chunk in text_content.split("\n\n") 
            if len(chunk.strip()) >= min_chunk_size
        ]
        
        if not chunks:
            st.warning(f"⚠️ No text chunks found with at least {min_chunk_size} characters.")
        else:
            with st.spinner(f"🔎 Embedding and storing {len(chunks)} text chunks..."):
                try:
                    vectors = embedder.encode(chunks, show_progress_bar=True).tolist()
                    points = [
                        PointStruct(id=str(uuid.uuid4()), vector=vec, payload={"text": chunk})
                        for vec, chunk in zip(vectors, chunks)
                    ]
                    qdrant.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
                    st.success(f"✅ {len(chunks)} chunks have been successfully embedded and stored!")
                except Exception as e:
                    st.error(f"❌ Error during embedding or storing: {e}")

# --- UI FOR QUERYING ---
st.header("2. Ask a Question")
user_query = st.text_input("Enter your question based on the document:")

if st.button("Get RAG Answer", disabled=not user_query.strip()):
    with st.spinner("🔍 Searching for relevant context..."):
        try:
            query_vec = embedder.encode(user_query).tolist()
            search_results = qdrant.search(
                collection_name=COLLECTION_NAME, 
                query_vector=query_vec, 
                limit=5  # Retrieve top 5 most relevant chunks
            )
            
            retrieved_chunks = [hit.payload['text'] for hit in search_results]
            context = "\n\n---\n\n".join(retrieved_chunks)

            st.subheader("📚 Retrieved Context")
            st.info(context)

            prompt = f"Based *only* on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {user_query}"
            
            with st.spinner("🤖 Generating answer with GPT-4o..."):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions strictly based on the provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                answer = response.choices[0].message.content.strip()

            st.subheader("💬 RAG Answer")
            st.success(answer)

        except Exception as e:
            st.error(f"❌ An error occurred during the RAG process: {e}")

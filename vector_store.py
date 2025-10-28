import os, tempfile
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb, config


def _client():
    return chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

def _split_docs(paths):
    docs = []
    for p in paths:
        try:
            docs += TextLoader(p).load()
        except:
            pass
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def cleanup_stale_collections():
    client = _client()
    for c in client.list_collections():
        if c.name.startswith(config.USER_COLLECTION_PREFIX):
            try:
                client.delete_collection(c.name)
                print(f"🧹 Removed stale collection: {c.name}")
            except Exception as e:
                print(f"⚠️ Error removing {c.name}: {e}")


def initialize_default_docs(api_key):
    client = _client()
    try:
        client.get_collection(config.DEFAULT_COLLECTION)
        return
    except:
        pass
    base = os.path.join(os.path.dirname(__file__), "default_docs")
    if not os.path.exists(base): return
    files = [os.path.join(base, f) for f in os.listdir(base) if f.endswith(".txt")]
    if not files: return
    docs = _split_docs(files)
    emb = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model=config.GEMINI_EMBEDDING_MODEL)
    Chroma.from_documents(docs, emb, collection_name=config.DEFAULT_COLLECTION, persist_directory=config.CHROMA_DB_PATH)

def load_user_docs(uploaded, api_key, session_id):
    client = _client()
    name = f"{config.USER_COLLECTION_PREFIX}{session_id}"
    try: client.delete_collection(name)
    except: pass

    paths = []
    for f in uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(f.getvalue()); paths.append(tmp.name)

    docs = _split_docs(paths)
    emb = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model=config.GEMINI_EMBEDDING_MODEL)
    Chroma.from_documents(docs, emb, collection_name=name, persist_directory=config.CHROMA_DB_PATH)
    for p in paths: os.remove(p)

def get_retriever(name, api_key):
    emb = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model=config.GEMINI_EMBEDDING_MODEL)
    store = Chroma(collection_name=name, persist_directory=config.CHROMA_DB_PATH, embedding_function=emb)
    return store.as_retriever(search_kwargs={"k": 3})

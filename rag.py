import os
import shutil
import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

def data_fingerprint() -> str:
    """data/ içeriği değişince farklı DB klasörü kullan."""
    h = hashlib.sha256()
    for p in sorted(DATA_DIR.glob("*")):
        if p.suffix.lower() in [".pdf", ".txt"] and p.is_file():
            st = p.stat()
            h.update(p.name.encode("utf-8"))
            h.update(str(st.st_size).encode("utf-8"))
            h.update(str(int(st.st_mtime)).encode("utf-8"))
    return h.hexdigest()[:10]

DB_DIR = BASE_DIR / f"chroma_db_gemini_{data_fingerprint()}"

def load_documents():
    docs = []
    for p in DATA_DIR.iterdir():
        if p.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
        elif p.suffix.lower() == ".txt":
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
    return docs

def build_db(embeddings):
    documents = load_documents()
    if not documents:
        raise RuntimeError("data/ klasöründe pdf/txt yok.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )
    db.persist()
    return db

def get_or_build_db(embeddings):
    if DB_DIR.exists() and any(DB_DIR.iterdir()):
        try:
            return Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
        except Exception:
            # DB bozuksa tamamen silip yeniden kur
            shutil.rmtree(DB_DIR, ignore_errors=True)

    return build_db(embeddings)

def create_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    db = get_or_build_db(embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 6})


    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY yok. Ortam değişkeni olarak ekle.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=api_key,
    )

    def answer(question: str) -> str:
        try:
            docs = retriever.invoke(question)
        except Exception:
            # Sorgu anında da bozulmuşsa: DB'yi yeniden kurup 1 kez daha dene
            shutil.rmtree(DB_DIR, ignore_errors=True)
            db2 = build_db(embeddings)
            retriever2 = db2.as_retriever(search_kwargs={"k": 6})
            docs = retriever2.invoke(question)

        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""Aşağıdaki BAĞLAM'a dayanarak soruyu cevapla.
Cevap bağlamda yoksa: "Bilmiyorum." yaz.
Cevabı Türkçe yaz.

BAĞLAM:
{context}

SORU:
{question}

CEVAP:"""

        return llm.invoke(prompt).content

    return answer

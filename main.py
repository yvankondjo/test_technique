import streamlit as st
from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.vector_store import VectorStore
from src.rag import RAGEngine
from src.db.database import Database
from src.ingest import DocumentExtractor
from src.chunking import DocumentChunker
from ui.monitoring import show_monitoring_page
from ui.documents import show_documents_page
from ui.chat import show_chat_page

st.set_page_config(
    page_title="RAG Juridique - Cabinet Parenti × AI Sister Test Technique",
    page_icon="⚖️",
    layout="wide"
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
    st.session_state.database = Database()
    st.session_state.rag_engine = RAGEngine(st.session_state.vector_store, st.session_state.database)
    st.session_state.extractor = DocumentExtractor()
    st.session_state.chunker = DocumentChunker(chunk_size=1024, chunk_overlap=128, context=True)

vector_store = st.session_state.vector_store
database = st.session_state.database
rag_engine = st.session_state.rag_engine
extractor = st.session_state.extractor
chunker = st.session_state.chunker

def main():
    if "page" not in st.session_state:
        st.session_state.page = "chat"
    
    with st.sidebar:
        st.markdown("## ⚖️ RAG Juridique - AI Sister Test Technique")
        st.markdown("---")
        
        pages = [
            {"icon": "💬", "name": "Chat", "key": "chat"},
            {"icon": "📁", "name": "Documents", "key": "documents"},
            {"icon": "📊", "name": "Monitoring", "key": "monitoring"}
        ]
        
        for page in pages:
            is_active = st.session_state.page == page["key"]
            button_type = "primary" if is_active else "secondary"
            
            if st.button(
                f"{page['icon']} {page['name']}",
                key=f"nav_{page['key']}",
                use_container_width=True,
                type=button_type
            ):
                st.session_state.page = page["key"]
                st.rerun()
    
    if st.session_state.page == "documents":
        show_documents_page(vector_store, database, extractor, chunker)
        return
    
    if st.session_state.page == "monitoring":
        show_monitoring_page(database)
        return
    
    show_chat_page(rag_engine, database, vector_store)

if __name__ == "__main__":
    main()


import streamlit as st
from src.rag import RAGEngine
from src.db.database import Database

def show_chat_page(rag_engine: RAGEngine, database: Database, vector_store=None):
    st.title("⚖️ RAG Juridique - Cabinet Parenti × AI Sister Test Technique")
    
    all_docs = database.get_all_documents()
    
    header_col1, header_col2 = st.columns([3, 1])
    
    with header_col1:
        st.subheader("💬 Chat avec l'assistant juridique")
    
    with header_col2:
        if st.button("🕐 Historique", key="history_button", use_container_width=True):
            st.session_state.show_history = not st.session_state.get('show_history', False)
            st.rerun()
    
    if st.session_state.get('show_history', False):
        with st.expander("📜 Historique des conversations", expanded=True):
            conversations = database.get_all_conversations()
            
            col_new, col_list = st.columns([1, 4])
            with col_new:
                if st.button("➕ Nouvelle conversation", key="new_conv_from_history", use_container_width=True):
                    st.session_state.conversation_id = None
                    st.session_state.messages = []
                    st.session_state.show_history = False
                    st.rerun()
            
            if conversations:
                for conv in conversations:
                    col_title, col_delete = st.columns([4, 1])
                    with col_title:
                        title = conv.title or f"Conversation #{conv.id}"
                        created = conv.created_at.strftime("%d/%m/%Y %H:%M") if conv.created_at else ""
                        is_selected = st.session_state.get('conversation_id') == conv.id
                        
                        if st.button(
                            f"{'▶️' if is_selected else '📝'} {title} - {created}",
                            key=f"select_conv_{conv.id}",
                            use_container_width=True
                        ):
                            st.session_state.conversation_id = conv.id
                            db_messages = database.get_conversation_messages(conv.id)
                            import json
                            st.session_state.messages = []
                            for msg in db_messages:
                                sources = []
                                chunks = []
                                if msg.sources:
                                    try:
                                        sources = json.loads(msg.sources)
                                        if sources and vector_store:
                                            chunks = vector_store.get_chunks_by_ids(sources)
                                    except:
                                        sources = []
                                st.session_state.messages.append({
                                    "role": msg.role,
                                    "content": msg.content,
                                    "sources": sources,
                                    "chunks": chunks
                                })
                            st.session_state.show_history = False
                            st.rerun()
                    
                    with col_delete:
                        if st.button("🗑️", key=f"delete_conv_{conv.id}", help="Supprimer cette conversation"):
                            database.delete_conversation(conv.id)
                            if st.session_state.get('conversation_id') == conv.id:
                                st.session_state.conversation_id = None
                                st.session_state.messages = []
                            st.success(f"Conversation supprimée")
                            st.rerun()
            else:
                st.info("Aucune conversation. Posez une question pour commencer.")
    
    if all_docs:
        st.markdown("---")
        st.markdown("### 📚 Documents indexés")
        doc_cols = st.columns(min(len(all_docs), 4))
        for idx, doc in enumerate(all_docs[:4]):
            with doc_cols[idx % 4]:
                st.markdown(f"**{doc.file_name}**")
                st.caption(f"{doc.chunk_count} chunks • {doc.file_type.upper()}")
        if len(all_docs) > 4:
            st.caption(f"... et {len(all_docs) - 4} autre(s) document(s)")
        st.markdown("---")
    else:
        st.warning("⚠️ Aucun document n'est actuellement indexé. Allez dans la page 'Documents' pour uploader et indexer des documents.")
    
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message.get("role") == "assistant" and message.get("details"):
                    with st.expander("📊 Détails"):
                        details = message["details"]
                        st.write(f"⏱️ Temps retrieval: {details.get('retrieval_time', 0):.2f}s")
                        st.write(f"⏱️ Temps génération: {details.get('generation_time', 0):.2f}s")
                        st.write(f"📎 Sources utilisées: {details.get('sources_count', 0)} chunks")
                
                if message.get("sources"):
                    with st.expander(f"📎 Sources utilisées ({len(message['sources'])} chunks)"):
                        sources_info = message.get("chunks", [])
                        if sources_info:
                            for idx, chunk_info in enumerate(sources_info[:10], 1):
                                metadata = chunk_info.get('metadata', {})
                                doc_name = metadata.get('filename', 'Document inconnu')
                                doc_id = metadata.get('document_id', 'N/A')
                                chunk_idx = metadata.get('chunk_index', 'N/A')
                                chunk_text = chunk_info.get('text', '')[:150]
                                
                                st.markdown(f"**Source {idx}:** {doc_name}")
                                st.caption(f"Document ID: {doc_id} | Chunk #{chunk_idx}")
                                if chunk_text:
                                    st.text(f"{chunk_text}...")
                                st.markdown("---")
                            
                            if len(sources_info) > 10:
                                st.caption(f"... et {len(sources_info) - 10} autre(s) source(s)")
                        else:
                            st.write(f"{len(message['sources'])} chunks référencés")
    
    if prompt := st.chat_input("Posez votre question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Recherche dans les documents..."):
                try:
                    result = rag_engine.chat(
                        message=prompt,
                        conversation_id=st.session_state.conversation_id
                    )
                    
                    st.session_state.conversation_id = result["conversation_id"]
                    answer = result["answer"]
                    st.markdown(answer)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": result.get("sources", []),
                        "chunks": result.get("chunks", []),
                        "details": {
                            "retrieval_time": result.get("retrieval_time", 0),
                            "generation_time": result.get("generation_time", 0),
                            "sources_count": len(result.get("sources", []))
                        }
                    })
                    st.rerun()
                except Exception as e:
                    error_msg = f"❌ Erreur lors de la génération de la réponse: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

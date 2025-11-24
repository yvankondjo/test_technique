import streamlit as st
from pathlib import Path
from src.config import Config
from src.db.schema import Document

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

def validate_file_size(file) -> tuple[bool, str]:
    if file.size > MAX_FILE_SIZE_BYTES:
        return False, f"Fichier trop volumineux ({file.size / 1024 / 1024:.1f} MB). Maximum: {MAX_FILE_SIZE_MB} MB"
    return True, ""

def show_documents_page(vector_store, database, extractor, chunker):
    st.title("📁 Gestion des Documents")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload de documents")
        
        uploaded_file = st.file_uploader(
            "Sélectionner un document à indexer",
            type=['txt', 'pdf', 'docx', 'html', 'csv'],
            help=f"Formats supportés: TXT, PDF, DOCX, HTML, CSV (max {MAX_FILE_SIZE_MB} MB)"
        )
        
        if uploaded_file is not None:
            is_valid, error_msg = validate_file_size(uploaded_file)
            if not is_valid:
                st.error(error_msg)
            else:
                st.info(f"📄 {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
                
                if st.button("📤 Traiter et indexer", type="primary"):
                    with st.spinner("Traitement en cours..."):
                        try:
                            save_path = Config.SAVE_DOCUMENTS_DIR / uploaded_file.name
                            save_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            with open(save_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            result = extractor.extract_file(str(save_path))
                            
                            if "error" in result:
                                st.error(f"❌ Erreur d'extraction: {result['error']}")
                            else:
                                content = result.get("content", "")
                                if not content or len(content.strip()) == 0:
                                    st.warning("⚠️ Le document est vide ou n'a pas pu être extrait")
                                else:
                                    extension = Path(uploaded_file.name).suffix.lower()
                                    
                                    with st.spinner("Découpage en chunks..."):
                                        if extension == ".csv":
                                            chunks = chunker.chunk_csv(content)
                                        else:
                                            chunks = chunker.chunk_text(content)
                                    
                                    if not chunks:
                                        st.warning("⚠️ Aucun chunk généré")
                                    else:
                                        import uuid
                                        base_name = Path(uploaded_file.name).stem
                                        document_id = f"{base_name}_{uuid.uuid4().hex[:8]}"
                                        doc_metadata = {
                                            "document_id": document_id,
                                            "file_path": str(save_path),
                                            "filename": uploaded_file.name,
                                            "file_type": extension.replace(".", "")
                                        }
                                        
                                        with st.spinner("Indexation dans la base vectorielle..."):
                                            vector_store.add_documents(doc_metadata, chunks)
                                        
                                        database.save_document(
                                            document_id=document_id,
                                            file_path=str(save_path),
                                            file_name=uploaded_file.name,
                                            file_type=extension.replace(".", ""),
                                            chunk_count=len(chunks)
                                        )
                                        
                                        st.success(f"✅ Document indexé avec succès: {len(chunks)} chunks")
                                        st.rerun()
                                        
                        except Exception as e:
                            st.error(f"❌ Erreur: {str(e)}")
                            st.exception(e)
    
    with col2:
        st.subheader("📊 Statistiques")
        all_docs = database.get_all_documents()
        total_docs = len(all_docs) if all_docs else 0
        total_chunks = sum(doc.chunk_count for doc in all_docs) if all_docs else 0
        
        st.metric("Documents indexés", total_docs)
        st.metric("Chunks total", total_chunks)
        
    
    st.markdown("---")
    st.subheader("📚 Documents indexés")
    
    all_docs = database.get_all_documents()
    if not all_docs:
        st.info("ℹ️ Aucun document indexé. Utilisez le formulaire ci-dessus pour en ajouter.")
    else:
        for doc in all_docs:
            with st.container():
                col1, col2, col3 = st.columns([4, 1, 1])
                with col1:
                    st.markdown(f"**📄 {doc.file_name}**")
                    st.caption(f"ID: {doc.document_id} | {doc.chunk_count} chunks | {doc.file_type.upper()}")
                    if doc.file_path:
                        st.caption(f"📍 {doc.file_path}")
                
                with col2:
                    if st.button("🔄 Ré-indexer", key=f"reindex_{doc.id}"):
                        with st.spinner("Ré-indexation en cours..."):
                            try:
                                if doc.file_path and Path(doc.file_path).exists():
                                    vector_store.delete_document(doc.document_id)
                                    
                                    result = extractor.extract_file(doc.file_path)
                                    
                                    if "error" in result:
                                        st.error(f"❌ Erreur d'extraction: {result['error']}")
                                    else:
                                        content = result.get("content", "")
                                        if content and len(content.strip()) > 0:
                                            extension = Path(doc.file_path).suffix.lower()
                                            
                                            if extension == ".csv":
                                                chunks = chunker.chunk_csv(content)
                                            else:
                                                chunks = chunker.chunk_text(content)
                                            
                                            if chunks:
                                                doc_metadata = {
                                                    "document_id": doc.document_id,
                                                    "file_path": doc.file_path,
                                                    "filename": doc.file_name,
                                                    "file_type": doc.file_type
                                                }
                                                vector_store.add_documents(doc_metadata, chunks)
                                                
                                                database.update_document_chunk_count(doc.document_id, len(chunks))
                                                
                                                st.success(f"✅ Document ré-indexé: {len(chunks)} chunks")
                                                st.rerun()
                                            else:
                                                st.warning("⚠️ Aucun chunk généré")
                                        else:
                                            st.warning("⚠️ Le document est vide")
                                else:
                                    st.error("❌ Fichier source introuvable")
                            except Exception as e:
                                st.error(f"❌ Erreur lors de la ré-indexation: {str(e)}")
                                st.exception(e)
                
                with col3:
                    delete_key = f"delete_{doc.id}"
                    if delete_key not in st.session_state:
                        st.session_state[delete_key] = False
                    
                    if not st.session_state[delete_key]:
                        if st.button("🗑️ Supprimer", key=f"btn_{doc.id}", type="secondary"):
                            st.session_state[delete_key] = True
                            st.rerun()
                    else:
                        st.warning(f"⚠️ Confirmer la suppression de '{doc.file_name}' ?")
                        col_confirm, col_cancel = st.columns(2)
                        with col_confirm:
                            if st.button("✅ Confirmer", key=f"confirm_{doc.id}", type="primary"):
                                try:
                                    file_name = doc.file_name if doc and getattr(doc, 'file_name', None) else "<document>"

                                    vector_store.delete_document(doc.document_id)

                                    if doc.file_path and Path(doc.file_path).exists():
                                        try:
                                            Path(doc.file_path).unlink()
                                        except Exception as file_err:
                                            st.warning(f"⚠️ Fichier source non supprimé: {file_err}")

                                    database.session.query(Document).filter(Document.id == doc.id).delete()
                                    database.session.commit()
                                    st.session_state[delete_key] = False
                                    st.success(f"✅ Document '{file_name}' supprimé")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Erreur lors de la suppression: {str(e)}")
                                    st.session_state[delete_key] = False
                        with col_cancel:
                            if st.button("❌ Annuler", key=f"cancel_{doc.id}"):
                                st.session_state[delete_key] = False
                                st.rerun()
                
                st.markdown("---")


import streamlit as st
import pandas as pd
import plotly.express as px
from src.db.schema import RAGLog
import json

def show_monitoring_page(database):
    st.title("📊 Monitoring RAG")
    st.markdown("---")
    
    logs = database.session.query(RAGLog).order_by(RAGLog.timestamp.desc()).all()
    
    if not logs:
        st.info("Aucun log disponible pour le moment.")
        return
    
    df = pd.DataFrame([{
        'timestamp': log.timestamp,
        'conversation_id': log.conversation_id,
        'query': log.query[:100] + '...' if len(log.query) > 100 else log.query,
        'answer': log.answer[:100] + '...' if len(log.answer) > 100 else log.answer,
        'input_tokens': log.input_tokens,
        'output_tokens': log.output_tokens,
        'total_tokens': log.total_tokens,
        'retrieval_time': log.retrieval_time / 1000,
        'generation_time': log.generation_time / 1000,
        'total_time': log.total_time / 1000,
        'context_length': log.context_length,
        'sources_count': len(json.loads(log.sources)) if log.sources else 0
    } for log in logs])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total requêtes", len(logs))
    with col2:
        avg_time = df['total_time'].mean()
        st.metric("Temps moyen", f"{avg_time:.2f}s")
    with col3:
        avg_tokens = df['total_tokens'].mean()
        st.metric("Tokens moyens", f"{avg_tokens:.0f}")
    with col4:
        total_tokens = df['total_tokens'].sum()
        st.metric("Total tokens", f"{total_tokens:,}")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📈 Graphiques", "📋 Logs détaillés", "🔍 Analyse"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Temps de réponse")
            fig_time = px.line(
                df, 
                x='timestamp', 
                y=['retrieval_time', 'generation_time', 'total_time'],
                labels={'value': 'Temps (s)', 'timestamp': 'Date'},
                title="Évolution des temps de réponse"
            )
            st.plotly_chart(fig_time, width='stretch')
        
        with col2:
            st.subheader("Tokens utilisés")
            fig_tokens = px.bar(
                df.tail(20),
                x='timestamp',
                y=['input_tokens', 'output_tokens'],
                labels={'value': 'Tokens', 'timestamp': 'Date'},
                title="Tokens par requête (20 dernières)"
            )
            st.plotly_chart(fig_tokens, width='stretch')
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Distribution temps total")
            fig_dist = px.histogram(
                df,
                x='total_time',
                nbins=20,
                labels={'total_time': 'Temps total (s)', 'count': 'Fréquence'},
                title="Distribution des temps de réponse"
            )
            st.plotly_chart(fig_dist, width='stretch')
        
        with col4:
            st.subheader("Taille du contexte")
            fig_context = px.scatter(
                df.tail(50),
                x='context_length',
                y='total_time',
                labels={'context_length': 'Taille contexte (caractères)', 'total_time': 'Temps (s)'},
                title="Taille contexte vs Temps"
            )
            st.plotly_chart(fig_context, width='stretch')
    
    with tab2:
        st.subheader("Logs détaillés")
        
        search_query = st.text_input("🔍 Rechercher dans les logs", "")
        
        if search_query:
            filtered_df = df[df['query'].str.contains(search_query, case=False, na=False) | 
                            df['answer'].str.contains(search_query, case=False, na=False)]
        else:
            filtered_df = df
        
        st.dataframe(
            filtered_df[['timestamp', 'query', 'input_tokens', 'output_tokens', 'total_time', 'sources_count']],
            width='stretch',
            height=400
        )
    
    with tab3:
        st.subheader("Statistiques détaillées")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Temps**")
            st.write(f"- Retrieval moyen: {df['retrieval_time'].mean():.3f}s")
            st.write(f"- Génération moyen: {df['generation_time'].mean():.3f}s")
            st.write(f"- Total moyen: {df['total_time'].mean():.3f}s")
            st.write(f"- Total min: {df['total_time'].min():.3f}s")
            st.write(f"- Total max: {df['total_time'].max():.3f}s")
        
        with col2:
            st.write("**Tokens**")
            st.write(f"- Input moyen: {df['input_tokens'].mean():.0f}")
            st.write(f"- Output moyen: {df['output_tokens'].mean():.0f}")
            st.write(f"- Total moyen: {df['total_tokens'].mean():.0f}")
            st.write(f"- Total cumulé: {df['total_tokens'].sum():,}")
        
        st.write("**Contexte**")
        st.write(f"- Taille moyenne: {df['context_length'].mean():.0f} caractères")
        st.write(f"- Sources moyennes: {df['sources_count'].mean():.1f} chunks")
        
        if len(df) > 0:
            st.write("**Évolution**")
            recent_avg = df.tail(10)['total_time'].mean()
            older_avg = df.head(max(1, len(df) - 10))['total_time'].mean() if len(df) > 10 else recent_avg
            if older_avg > 0:
                improvement = ((older_avg - recent_avg) / older_avg) * 100
                st.write(f"- Amélioration récente: {improvement:+.1f}%")



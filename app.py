import streamlit as st
from pdf_processor import extract_text_from_pdf, split_text_into_chunks
from indexer import SemanticIndexer
from query import ResearchAssistant

st.title("AI Open Source Research Assistant")

st.markdown("Upload a PDF document, build a semantic index, and ask questions about its content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if 'indexer' not in st.session_state:
        with st.spinner("Processing PDF and building semantic index..."):
            # Extract text from the uploaded file
            text = extract_text_from_pdf(uploaded_file)
            # Split text into chunks
            chunks = split_text_into_chunks(text)
            texts = [chunk['text'] for chunk in chunks]
            metadatas = [{'page': chunk['page'], 'chunk_id': chunk['chunk_id']} for chunk in chunks]
            # Build the index
            indexer = SemanticIndexer()
            indexer.build_index(texts, metadatas)
            st.session_state.indexer = indexer
        st.success("Index built successfully! You can now ask questions about the document.")
    else:
        st.info("Index already built from the uploaded PDF.")

    # Query input
    question = st.text_input("Ask a question about the document:")

    if question:
        if 'indexer' in st.session_state:
            with st.spinner("Generating answer..."):
                assistant = ResearchAssistant(st.session_state.indexer)
                result = assistant.answer_question(question)
            st.subheader("Answer:")
            st.write(result['answer'])
            st.subheader("Supporting Evidence:")
            for evidence in result['evidence']:
                st.write(f"**Page {evidence['page']}:** {evidence['text']}")
        else:
            st.error("Please upload a PDF first.")
else:
    st.info("Please upload a PDF file to get started.")

import hashlib
from pathlib import Path

import streamlit as st

from rag import (
    PDF_PATH,
    extract_pages,
    create_chunks,
    create_embeddings,
    build_vector_index,
    search_chunks,
    generate_answer,
)


# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Document Assistant")
st.caption(
    "Upload a PDF, ask questions, and get answers "
    "with document source references."
)


# =========================
# PDF UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"],
)

active_pdf_path = PDF_PATH

if uploaded_file is not None:
    try:
        pdf_bytes = uploaded_file.getvalue()

        content_hash = hashlib.sha256(
            pdf_bytes
        ).hexdigest()

        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        safe_filename = Path(uploaded_file.name).name
        active_pdf_path = str(upload_dir / safe_filename)

        Path(active_pdf_path).write_bytes(pdf_bytes)

    except OSError as error:
        st.error(f"Could not save the uploaded PDF: {error}")
        st.stop()

else:
    try:
        pdf_bytes = Path(PDF_PATH).read_bytes()
        content_hash = hashlib.sha256(pdf_bytes).hexdigest()

    except OSError as error:
        st.error(f"Could not read the default PDF: {error}")
        st.stop()


# =========================
# PROCESS AND INDEX PDF
# =========================

@st.cache_resource(show_spinner=False)
def load_rag_system(pdf_path, document_hash):
    pages = extract_pages(pdf_path)

    if not pages:
        raise ValueError(
            "No readable text was found in this PDF."
        )

    chunks = create_chunks(pages)

    if not chunks:
        raise ValueError(
            "No text chunks were created."
        )

    chunks = create_embeddings(chunks)
    index = build_vector_index(chunks)

    return pages, chunks, index


try:
    with st.spinner("Preparing your PDF for search..."):
        pages, chunks, index = load_rag_system(
            active_pdf_path,
            content_hash,
        )

except Exception as error:
    st.error(f"Could not process the PDF: {error}")
    st.stop()


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.header("Document Details")
    st.write("**File:**", Path(active_pdf_path).name)

    st.metric("Pages", len(pages))
    st.metric("Text chunks", len(chunks))

    st.divider()

    st.write("**Embedding model**")
    st.code("nomic-embed-text")

    st.write("**Language model**")
    st.code("llama3.2:3b")

    st.caption("Both models run locally using Ollama.")


# =========================
# QUESTION FORM
# =========================

with st.form("question_form"):
    question = st.text_input(
        "Ask a question about your document",
        placeholder=(
            "How many days per week can employees "
            "work from home?"
        ),
    )

    submitted = st.form_submit_button(
        "Find Answer",
        type="primary",
    )


# =========================
# RETRIEVE AND GENERATE
# =========================

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:
        with st.spinner(
            "Searching the document and generating an answer..."
        ):
            results = search_chunks(
                question=question,
                chunks=chunks,
                index=index,
                top_k=3,
            )

            answer = generate_answer(
                question=question,
                retrieved_chunks=results,
            )

        # Show the answer
        st.subheader("Answer")
        st.write(answer)

        # Detect the expected not-found responses
        normalized_answer = answer.casefold()

        answer_not_found = (
            "could not find this information in the document"
            in normalized_answer
            or "could not find relevant information"
            in normalized_answer
        )

        # Show sources only when an answer was found
        if answer_not_found or not results:
            st.info(
                "No supporting source was found for this answer."
            )

        else:
            st.subheader("Sources")

            seen_sources = set()

            for result in results:
                source_key = (
                    result["source"],
                    result["page"],
                )

                if source_key not in seen_sources:
                    st.write(
                        f"📄 **{Path(result['source']).name}** "
                        f"— Page {result['page']}"
                    )

                    seen_sources.add(source_key)

            # Display retrieved passages only for answered questions
            with st.expander("View retrieved passages"):
                for number, result in enumerate(results, start=1):
                    st.markdown(f"**Passage {number}**")

                    st.caption(
                        f"Page {result['page']} | "
                        f"Similarity: {result['score']:.3f}"
                    )

                    st.write(result["text"])
                    st.divider()

    except Exception as error:
        st.error(
            f"Could not generate the answer: {error}"
        )
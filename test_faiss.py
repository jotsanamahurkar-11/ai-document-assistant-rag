from rag import (
    PDF_PATH,
    extract_pages,
    create_chunks,
    create_embeddings,
    build_vector_index,
    search_chunks,
)

# Step 1: Read the PDF
pages = extract_pages(PDF_PATH)

# Step 2: Split text into chunks
chunks = create_chunks(pages)

# Step 3: Create embeddings
chunks = create_embeddings(chunks)

# Step 4: Store embeddings in FAISS
index = build_vector_index(chunks)

# Step 5: Search for relevant information
question = "How many days per week can employees work from home?"

results = search_chunks(
    question=question,
    chunks=chunks,
    index=index,
    top_k=3,
)

# Step 6: Display search results
print("\nQuestion:", question)
print("\nRetrieved chunks:")

for result in results:
    print("\n" + "-" * 50)
    print("Source:", result["source"])
    print("Page:", result["page"])
    print("Similarity:", round(result["score"], 3))
    print("Text:", result["text"])
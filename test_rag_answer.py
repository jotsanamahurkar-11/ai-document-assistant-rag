from rag import (
    PDF_PATH,
    extract_pages,
    create_chunks,
    create_embeddings,
    build_vector_index,
    search_chunks,
    generate_answer,
)

# Step 1: Read the PDF
pages = extract_pages(PDF_PATH)

# Step 2: Split the text into chunks
chunks = create_chunks(pages)

# Step 3: Create embeddings
chunks = create_embeddings(chunks)

# Step 4: Build the FAISS index
index = build_vector_index(chunks)

# Step 5: Ask a question
question = "How many days per week can employees work from home?"

# Step 6: Retrieve relevant chunks
results = search_chunks(
    question=question,
    chunks=chunks,
    index=index,
    top_k=3,
)

# Step 7: Generate an answer using Llama
answer = generate_answer(question, results)

# Step 8: Display the result
print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(answer)

print("\nSOURCES:")
for result in results:
    print(f"- {result['source']} | Page {result['page']}")
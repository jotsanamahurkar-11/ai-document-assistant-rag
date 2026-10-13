from rag import PDF_PATH, extract_pages, create_chunks, create_embeddings

# Step 1: Extract PDF text
pages = extract_pages(PDF_PATH)

# Step 2: Create chunks
chunks = create_chunks(pages)

# Step 3: Create embeddings
embedded_chunks = create_embeddings(chunks)

# Step 4: Display results
print("Total pages:", len(pages))
print("Total chunks:", len(embedded_chunks))

if embedded_chunks:
    first_chunk = embedded_chunks[0]

    print("Embedding created successfully.")
    print("Vector dimensions:", len(first_chunk["embedding"]))
    print("Source:", first_chunk["source"])
    print("Page:", first_chunk["page"])
    print("First 5 vector values:", first_chunk["embedding"][:5])
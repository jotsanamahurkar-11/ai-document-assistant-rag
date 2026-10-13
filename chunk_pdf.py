import pymupdf


PDF_PATH = "data/employee_handbook.pdf"

CHUNK_SIZE = 100      # words per chunk
CHUNK_OVERLAP = 20    # overlapping words


def extract_pages(pdf_path):
    doc = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc):
        text = page.get_text("text").strip()

        if text:
            pages.append({
                "text": text,
                "page": page_number + 1,
                "source": pdf_path
            })

    doc.close()

    return pages


def create_chunks(pages):
    chunks = []

    for page in pages:
        words = page["text"].split()

        start = 0

        while start < len(words):
            end = start + CHUNK_SIZE

            chunk_text = " ".join(words[start:end])

            chunks.append({
                "text": chunk_text,
                "page": page["page"],
                "source": page["source"]
            })

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


pages = extract_pages(PDF_PATH)
chunks = create_chunks(pages)

print(f"Total pages: {len(pages)}")
print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print("\n" + "=" * 60)
    print(f"CHUNK {i}")
    print(f"PAGE: {chunk['page']}")
    print("=" * 60)
    print(chunk["text"])
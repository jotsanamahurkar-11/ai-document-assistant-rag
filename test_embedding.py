import ollama

response = ollama.embed(
    model="nomic-embed-text",
    input="Employees are entitled to 18 annual leave days."
)

embedding = response["embeddings"][0]

print("Embedding created successfully.")
print("Number of dimensions:", len(embedding))
print("First 5 values:", embedding[:5])
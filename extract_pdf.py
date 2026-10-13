import pymupdf

pdf_path = "data/employee_handbook.pdf"

doc = pymupdf.open(pdf_path)

for page_number, page in enumerate(doc):
    text = page.get_text("text")

    print(f"\n--- PAGE {page_number + 1} ---\n")
    print(text)

doc.close()
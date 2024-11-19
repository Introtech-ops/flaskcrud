import pdfplumber

# Function to load and extract text from the PDF
def load_building_code(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:  # Check if the page has extractable text
                    text += f"--- Page {page_number} ---\n{page_text}\n\n"
            return text
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return ""

# Test function to display extracted content
def test_pdf_handling(file_path):
    extracted_text = load_building_code(file_path)
    if extracted_text:
        print("Extracted text from the PDF:")
        print(extracted_text[:1000])  # Print the first 1000 characters for brevity
    else:
        print("No text extracted or an error occurred.")

# Path to the PDF file (adjust the path as necessary)
pdf_path = 'National_Building_Code_2024.pdf'

# Run the test
test_pdf_handling(pdf_path)

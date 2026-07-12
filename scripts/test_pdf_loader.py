import os
from langchain_community.document_loaders import PyPDFLoader

# Point this directly to the file you just downloaded
PDF_PATH = "../data/constitution_of_India.pdf"

def test_pdf_extraction(file_path):
    print(f"Attempting to load: {file_path}...\n")
    
    try:
        # Initialize the LangChain PDF Loader
        loader = PyPDFLoader(file_path)
        
        # Load the document into a list of "Pages"
        pages = loader.load()
        
        print(f"✅ Success! Successfully parsed {len(pages)} pages.")
        print("-" * 50)
        print("📄 Preview of Page 1 (First 500 characters):\n")
        
        # Print a snippet of the first page to verify it's reading text, not gibberish
        print(pages[0].page_content[:500])
        print("\n" + "-" * 50)
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file at {file_path}.")
        print("Please check that the file is in the 'data' folder and named correctly.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Ensure the script runs relative to where it is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_pdf_path = os.path.join(script_dir, PDF_PATH)
    
    test_pdf_extraction(absolute_pdf_path)
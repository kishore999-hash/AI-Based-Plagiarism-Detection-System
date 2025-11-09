import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def check_plagiarism(pdf1_path, pdf2_path):
    text1 = extract_text_from_pdf(pdf1_path)
    text2 = extract_text_from_pdf(pdf2_path)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors)[0][1]
    percentage = round(similarity * 100, 2)
    return percentage

if __name__ == "__main__":
    print("=== PDF Plagiarism Checker using NLP ===\n")
    pdf1_path = input("Enter path for first PDF file: ").strip()
    pdf2_path = input("Enter path for second PDF file: ").strip()
    try:
        result = check_plagiarism(pdf1_path, pdf2_path)
        print("\n-----------------------------------")
        print(f"Plagiarism Percentage: {result}%")
        if result > 80:
            print("⚠️ High similarity detected! (Possible Plagiarism)")
        elif result > 50:
            print("⚠️ Moderate similarity.")
        else:
            print("✅ Low similarity. Documents are mostly unique.")
    except Exception as e:
        print("\n❌ Error:", e)
        print("Please check your file paths or PDF format.")

from PyPDF2 import PdfReader
import spacy
import json

nlp = spacy.load("en_core_web_sm")
keywords = ["dataset","architecture"]

reader = PdfReader('teste1.pdf')
 
content = ''

# merge ao pdf content in one variable
for page in reader.pages:
    content += page.extract_text()

doc = nlp(content)
articleData = {}

# initialization keywords arrays
for keyword in keywords:
    articleData[keyword] = []

for sent in doc.sents:
    for keyword in keywords:
        if keyword in sent.text.lower():
            articleData[keyword].append(sent.text.strip())

with open('data_collection.json', 'w') as f:
    json.dump(articleData, f, indent=4)
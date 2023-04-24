from glob import glob
import re

def clean_text(text):
    # Remove odd newlines
    cleaned_text = re.sub(r'\n{2,}', '\n', text)
    
    # Remove unnecessary spaces
    cleaned_text = re.sub(r'\s\s+', ' ', cleaned_text)
    
    # Remove tabs
    cleaned_text = re.sub(r'\t', ' ', cleaned_text)
    
    cleaned_text = re.sub(r"[^a-zA-Z0-9.,!?'\-;:()\"\n\t\s]", "", cleaned_text)
    
    return cleaned_text.strip()

train = ""
text_batch = 'texts/' + input("batch: ") + '/*.txt'
for filepath in glob(text_batch):
    with open(filepath, 'r', encoding='utf-8') as f:
         text = clean_text(f.read())
    train += text
with open('train.txt', 'w', encoding = 'utf-8') as f:
    f.write(train)
from glob import glob
import re

def clean_text(text):
    # Remove odd newlines
    cleaned_text = re.sub(r'\n{2,}', '\n', text)
    
    # Remove unnecessary spaces
    cleaned_text = re.sub(r'\s\s+', ' ', cleaned_text)
    
    # Remove tabs
    cleaned_text = re.sub(r'\t', ' ', cleaned_text)
    
    return cleaned_text.strip()

train = ""
for filepath in glob("texts/*.txt"):
    with open(filepath, 'r', encoding='utf-8') as f:
         text = clean_text(f.read())
    train += text
with open('test_train.txt', 'w', encoding = 'utf-8') as f:
    f.write(train)
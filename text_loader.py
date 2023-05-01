from glob import glob
import re
#from datasets import load_dataset

def clean_text(text):
    # Remove odd newlines
    cleaned_text = re.sub(r'\n{2,}', '\n', text)
    # Remove unnecessary spaces
    cleaned_text = re.sub(r'\s\s+', ' ', cleaned_text)
    # Remove tabs
    cleaned_text = re.sub(r'\t', ' ', cleaned_text)
    

    return cleaned_text.strip()

train = ""
if input("openwebtext: y/n") == "y":
    
    dataset = load_dataset("openwebtext")
    train = dataset['train']
    limit = int(input("# of examples: "))
    with open('train.txt', 'w', encoding='utf-8') as f:
        for i in range(limit):
            f.write(train[i]['text'])
            f.write("\n")
            
    
else:
    text_batch = 'texts/' + input("batch: ") + '/*.txt'
    for filepath in glob(text_batch):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = clean_text(f.read())
        train += text
    with open('train.txt', 'w', encoding = 'utf-8') as f:
        f.write(train)
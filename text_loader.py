from glob import glob

train = ""
for filepath in glob("texts/*.txt"):
    with open(filepath, 'r', encoding='utf-8') as f:
         text = f.read()
    train += text
with open('test_train.txt', 'w', encoding = 'utf-8') as f:
    f.write(train)
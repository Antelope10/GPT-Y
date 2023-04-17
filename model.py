import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32
block_size = 8
max_iters = 2000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
#head_size = 16

torch.manual_seed(1337)

with open('train.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text),dtype=torch.long)
n = int(.9*len(data))
train_data = data[:n]
test_data = data[n:]

print(data.shape)

def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

def estimate_loss():
    out = {}
    m.eval()
    for split in ['train','test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb,yb = get_batch(split)
            logits,loss = m(xb,yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) #(B,T,head_size)
        q = self.query(x) #(B,T,head_size)
        v = self.value(x) #(B,T,head_size)
        
        wei = q @ k.transpose(-2,-1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) #preents past communication
        wei = F.softmax(wei,dim=-1)
        
        v = self.value(x)
        out = wei @ v
        return out        
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        
    def forward(self,x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class BigramLanguageModel(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.ln_head = nn.Linear(n_embd,vocab_size)
    self.sa_heads = MultiHeadAttention(4,int(n_embd/4))
  
  def forward(self, idx, targets=None):
    
    tok_emb = self.token_embedding_table(idx) #(B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(block_size,device=device)) #(T,C)
    
    x = tok_emb + pos_emb
    x = self.sa_heads(x) #(B,T,head_size)
    
    logits = self.ln_head(x) #(B,T,vocab_size)
    B,T,C = logits.shape
    logits = logits.view(B*T,C) #(batch_size, context window, vocab_size)
    
    loss = None
    if targets is not None:
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits,targets)

    logits = logits.view(B,T,C) 
    return logits, loss

  def generate(self,idx,max_new_tokens):
    for _ in range(max_new_tokens):
        idx = idx[:,-block_size:]#restricts tokens passed to block_size
        print(idx.shape)
        logits, loss = self(idx)
        logits = logits[:,-1,:]
        probs = F.softmax(logits,dim=-1)
        idx_next = torch.multinomial(probs,num_samples=1).view(1,1)
        idx = torch.cat((idx,idx_next),dim=1)
    return idx

m = BigramLanguageModel()

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):
  if iter % eval_interval == 0:
      losses = estimate_loss()
      print(f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")  
    
  xb,yb = get_batch('train')

  logits, loss = m(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()


idx = torch.zeros((1,1),dtype=torch.long)
print(decode(m.generate(idx,max_new_tokens=100)[0].tolist()))
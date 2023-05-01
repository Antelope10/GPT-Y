# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
import time

#hyperparameters
batch_size = 64
block_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 512
dropout = 0.2
n_heads = 4
n_layer = 4
divergence = 0.2

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
print("vocab size", vocab_size)

def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device),y.to(device)

def estimate_loss():
    out = {}
    m.to(device)
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
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) #(B,T,head_size)
        q = self.query(x) #(B,T,head_size)
        v = self.value(x) #(B,T,head_size)
        
        wei = q @ k.transpose(-2,-1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) #preents past communication
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out        
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net =  nn.Sequential(nn.Linear(n_embd,4*n_embd),
                    nn.ReLU(),
                    nn.Linear(4*n_embd,n_embd),
                    nn.Dropout(dropout),
                
        )
    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self,n_embd,n_heads):
        super().__init__()
        self.heads = MultiHeadAttention(n_heads, n_embd // n_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class Transformer(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.ln_head = nn.Linear(n_embd,vocab_size)
    self.blocks = nn.Sequential(*[Block(n_embd,n_heads) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.ln_head = nn.Linear(n_embd,vocab_size)
  
  def forward(self, idx, targets=None):
    
    tok_emb = self.token_embedding_table(idx) #(B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(block_size,device=device)) #(T,C)
    
    x = tok_emb + pos_emb
    x = self.blocks(x)
    x = self.ln_f(x)
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
    if list(idx.shape)[1] < block_size:
        idx = torch.cat((torch.zeros(1,block_size-list(idx.shape)[1],dtype=torch.long), idx),dim=1)
    idx = idx.to(device)
    for _ in range(max_new_tokens):
        idx = idx[:,-block_size:]#restricts tokens passed to block_size
        logits, loss = self(idx)
        logits = logits[:,-1,:]
        probs = F.softmax(logits,dim=-1)
        idx_next = torch.multinomial(probs,num_samples=1).view(1,1)
        idx = torch.cat((idx,idx_next),dim=1)
    return idx

# %%
m = Transformer()
m = m.to(device)

print(sum(p.numel() for p in m.parameters()), "parameters") #print parameter count

if input("load model: y/n") == "y": #load model
    checkpoint_in = 'checkpoints/' + input("checkpoint: ") + '.pth'
    m.load_state_dict(torch.load(checkpoint_in,map_location=torch.device('cpu'))) #load params
    print("loaded from " + checkpoint_in)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# %%
start = time.time()
for iter in range(max_iters):
  if iter % eval_interval == 0:
      losses = estimate_loss()
      print(f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")  
      if abs(losses['train'] - losses['test']) > divergence * losses['test']:
        print("aborted training due to overfitting")
        break
  xb,yb = get_batch('train')
  logits, loss = m(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
end = time.time()
print("time elapsed: ", end-start)

# %%
if input("save model: y/n") == 'y':
    checkpoint_out = 'checkpoints/' + input("checkpoint: ") + '.pth'
    torch.save(m.state_dict(), checkpoint_out) 
    print("saved to " + checkpoint_out)

# %%
def complete(prompt, max_tokens):
    idx = torch.tensor(encode(prompt)).view(1,len(prompt))
    return prompt + decode(m.generate(idx,max_new_tokens=max_tokens)[0].tolist())

prompt = input("prompt")
print(complete(prompt,100))



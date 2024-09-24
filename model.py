from tinygrad import Tensor
import by

class Embeddings():
    def __init__(self, vocab_size, embed_dim, max_len):
        self.tok_emb = by.Embedding(vocab_size, embed_dim)
        self.pos_emb = by.Embedding(max_len, embed_dim)
        self
    
    def __call__(self, x):
        pos = Tensor.arange(x.shape[0])
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(pos)
        return tok_emb + pos_emb

class SelfAttention(): # encoder aja
    def __init__(self, embed_dim, head_dim):
        self.w_q = by.Linear(embed_dim, head_dim)
        self.w_k = by.Linear(embed_dim, head_dim)
        self.w_v = by.Linear(embed_dim, head_dim)
        # w_o waktu abisnya concat di MHA aja
    
    def __call__(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        
        attn_score = q @ k.T / (q.shape[-1] ** 0.5)
        attn_weight = by.Softmax()(attn_score)
        
        context_vector = attn_weight @ v
        
        return context_vector

class MultiHeadAttention():
    def __init__(self, embed_dim, num_heads):
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.attn = [SelfAttention(embed_dim, self.head_dim) for _ in range(num_heads)]
        self.w_o = by.Linear(embed_dim, embed_dim)
    
    def __call__(self, x):
        context_vectors = []
        for a in self.attn:
            context_vectors.append(a(x))
        
        context_vectors = Tensor(context_vectors)
        context_vector = Tensor.cat(context_vectors, dim=-1)
        return self.w_o(context_vector)

class FFN():
    def __init__(self, embed_dim, up_scale=4):
        self.ffn = by.Sequential([
            by.Linear(embed_dim, embed_dim * up_scale),
            by.GeLU(),
            by.Linear(embed_dim * up_scale, embed_dim),
        ])
    
    def __call__(self, x):
        return self.ffn(x)

class BERTTapiBukanBERT():
    def __init__(self, vocab_size, embed_dim, max_len, num_heads, num_layers):
        self.embed = Embeddings(vocab_size, embed_dim, max_len)
        self.mha = [
            MultiHeadAttention(embed_dim, num_heads) for _ in range(num_layers)
        ]
        self.ffn = FFN(embed_dim)
        self.norm_1 = by.RMSNorm(1)
        self.norm_2 = by.RMSNorm(1)
    
    def __call__(self, x):
        x = self.embed(x)
        for mha in self.mha:
            x = mha(x)
            x = self.norm_1(x)
            x = self.ffn(x)
            x = self.norm_2(x)
        return x
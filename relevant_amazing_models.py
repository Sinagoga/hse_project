class MultiHeadAttention(nn.Module):
    def __init__(self, dim_embedds, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert dim_embedds % num_heads == 0
        
        self.dim_embedds = dim_embedds
        self.num_heads = num_heads
        self.d_k = dim_embedds // num_heads
        
        self.W_q = nn.Linear(dim_embedds, dim_embedds)
        self.W_k = nn.Linear(dim_embedds, dim_embedds)
        self.W_v = nn.Linear(dim_embedds, dim_embedds)
        self.W_o = nn.Linear(dim_embedds, dim_embedds)
        self.dropout = nn.Dropout(0)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(self.dropout(attention), V)
        return output
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
      batch_size, seq_length, _ = Q.size()

      Q = self.W_q(Q)
      K = self.W_k(K)
      V = self.W_v(V)

      Q = Q.view(batch_size, seq_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)
      K = K.view(batch_size, seq_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)
      V = V.view(batch_size, seq_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)


      attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
      output = self.W_o(self.combine_heads(attn_output))
      return output

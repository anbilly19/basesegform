import torch
import torch.nn as nn

class CrossAttentionSingleInput(nn.Module):
    """
    Cross-Attention with single input that returns both queries and context.
    Queries attend to context (image features).
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Separate projections for queries and keys/values
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, num_queries=None):
        """
        Args:
            x: Concatenated tensor of shape (B, N+M, C) where:
                - First N tokens are queries
                - Last M tokens are context (image features)
            num_queries: Number of query tokens (N). If None, splits evenly.
        
        Returns:
            Output tensor of shape (B, N+M, C) - returns both attended queries and original context
        """
        B, total_tokens, C = x.shape
        
        if num_queries is None:
            num_queries = total_tokens // 2
        
        # Split into queries and context
        queries = x[:, :num_queries, :]  # (B, N, C)
        context = x[:, num_queries:, :]  # (B, M, C)
        
        N = num_queries
        M = total_tokens - num_queries
        
        # Project queries
        q = (
            self.q(queries)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        
        # Project keys and values from context
        kv = (
            self.kv(context)
            .reshape(B, M, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]
        
        # Cross-attention: queries attend to context
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to get updated queries
        updated_queries = (attn @ v).transpose(1, 2).reshape(B, N, C)
        updated_queries = self.proj(updated_queries)
        updated_queries = self.proj_drop(updated_queries)
        
        # Concatenate updated queries with original context
        output = torch.cat([updated_queries, context], dim=1)  # (B, N+M, C)
        
        return output
    
class TwoWayCrossAttentionSingleInput(nn.Module):
    """
    Two-way cross-attention with single input that returns both queries and context.
    Queries attend to context, and context attends to queries, with residual connections.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Projections for queries attending to context
        self.q_queries = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_context = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        # Projections for context attending to queries
        self.q_context = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_queries = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, num_queries=None):
        """
        Args:
            x: Concatenated tensor of shape (B, N+M, C) where:
                - First N tokens are queries
                - Last M tokens are context (image features)
            num_queries: Number of query tokens (N). If None, splits evenly.
        
        Returns:
            Output tensor of shape (B, N+M, C) - returns both attended queries and attended context
        """
        B, total_tokens, C = x.shape
        
        if num_queries is None:
            num_queries = total_tokens // 2
        
        # Split into queries and context
        queries = x[:, :num_queries, :]  # (B, N, C)
        context = x[:, num_queries:, :]  # (B, M, C)
        
        N = num_queries
        M = total_tokens - num_queries
        
        # First direction: queries attend to context
        q1 = (
            self.q_queries(queries)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        
        kv1 = (
            self.kv_context(context)
            .reshape(B, M, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k1, v1 = kv1[0], kv1[1]
        
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        
        attended_queries = (attn1 @ v1).transpose(1, 2).reshape(B, N, C)
        attended_queries = self.proj(attended_queries)
        attended_queries = self.proj_drop(attended_queries)
        attended_queries = attended_queries + queries  # Residual connection
        
        # Second direction: context attends to queries
        q2 = (
            self.q_context(context)
            .reshape(B, M, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        
        kv2 = (
            self.kv_queries(attended_queries)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k2, v2 = kv2[0], kv2[1]
        
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        
        attended_context = (attn2 @ v2).transpose(1, 2).reshape(B, M, C)
        attended_context = self.proj(attended_context)
        attended_context = self.proj_drop(attended_context)
        attended_context = attended_context + context  # Residual connection
        
        # Concatenate attended queries with attended context
        output = torch.cat([attended_queries, attended_context], dim=1)  # (B, N+M, C)
        
        return output
import mlx.core as mx
import mlx.nn as nn
from mlxim.model import create_model


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.head_dim = dim // n_heads
        self.n_heads = n_heads
        self.QKV = nn.Linear(dim, 3*dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.layernorm = nn.LayerNorm(dim)

    def __call__(self, X):
        # X = [b, t, dim]
        b, t, dim = X.shape
        X = self.layernorm(X)
        qkv = self.QKV(X)
        query, key, value = mx.split(qkv, 3, axis=-1)
        key = key.reshape(b, t, self.n_heads, self.head_dim)     # b, t, n_heads, head_dim
        query = query.reshape(b, t, self.n_heads, self.head_dim)   # b, t, n_heads, head_dim
        value = value.reshape(b, t, self.n_heads, self.head_dim)  # b, t, n_heads, head_dim

        attention_map = mx.einsum("bqnd,bknd->bnqk", query, key) / mx.sqrt(self.head_dim)
        attention_scores = mx.softmax(attention_map, axis=-1)

        # q = k = t
        # n = n_head

        attended = mx.einsum("bnqk,bknd->bnqd", attention_scores, value)

        # bnqd -> b, q, n, d
        # bntd -> b, t, n, d
        attended_t = attended.transpose(0, 2, 1, 3)
    
        # concat over n_heads
        concat_attended_t = attended_t.reshape(b, t, dim)

        output = self.out(concat_attended_t)

        return output


class MLXVideoClassifier(nn.Module):
    def __init__(self,
        num_classes: int,
        num_frames_per_video: int,
        dim: int,
        n_heads: int,
    ):
        super().__init__()
        self.resnet = create_model("resnet18", weights=True, num_classes=1000)
        self.W = nn.Linear(512, dim)
        self.dim = dim
        self.n_heads = n_heads
        self.attention = MultiHeadAttention(dim, n_heads)
        self.classifier = nn.Linear(dim, num_classes)
        self.cls = mx.random.normal([dim])
        self.layernorm = nn.LayerNorm(dim)

    def __call__(self, X):
        # X is a [b, t, H, W, C]
        b, t, _, _, _ = X.shape
        X_bt = X.reshape((b * t, *X.shape[2:]))
        # b*t, dim
        frame_activations = self.W(self.resnet.get_features(X_bt))  # b*t, dim

        # b, t, dim
        frame_activations_t = frame_activations.reshape(b, t, -1)

        # b, 1, dim
        cls_b = mx.tile(self.cls, [b, 1, 1])

        # b, t+1, dim
        frame_activations_with_cls = mx.concat([cls_b, frame_activations_t], axis=1)

        attended_frames = self.attention(frame_activations_with_cls)

        cls_activations = attended_frames[:, 0, :]
        attended_frames = self.layernorm(cls_activations)

        logits = mx.softmax(self.classifier(cls_activations), axis=1)
        
        return logits

import torch
import torch.nn as nn

class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, feed_forward_dim, dropout_rate=0.1):
        super(ConformerBlock, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, d_model),
            nn.Dropout(dropout_rate)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        attended = self.multihead_attention(x, x, x)[0]
        x = residual + self.dropout(attended)

        residual = x
        x = self.layer_norm2(x)
        feed_forward_out = self.feed_forward(x)
        x = residual + self.dropout(feed_forward_out)
        return x

class Conformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, feed_forward_dim, num_classes, dropout_rate=0.1):
        super(Conformer, self).__init__()
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, feed_forward_dim, dropout_rate) 
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.size = d_model
     #    self.output_linear = nn.Linear(d_model, num_classes)

    def forward(self, x):
        for conformer_block in self.conformer_blocks:
            x = conformer_block(x)
        x = self.layer_norm(x)
     #    x = torch.mean(x, dim=1)  # Global average pooling
     #    x = self.output_linear(x)
        return x


if __name__ == '__main__':
    input_dim = 80  # Dimensionality of the input features
    seq_length = 30  # Length of the input sequence
    batch_size = 16  # Number of sequences in a batch
    num_classes = 10  # Number of output classes

    # Generate random input tensor
    input_tensor = torch.randn(batch_size, seq_length, input_dim)

    # Create an instance of the Conformer model
    model = Conformer(d_model=80, num_heads=4, num_layers=6, feed_forward_dim=1024, num_classes=num_classes)

    # Forward pass
    output = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
class ModelNew(Pooler):
    def __init__(self, hidden_size):
        super().__init__(hidden_size)
    def forward(self, hidden_states, sequence_index=0):
        pooled = hidden_states[:, sequence_index, :]
        # use custom kernel
        out = triton_linear_tanh(pooled, self.dense.weight, self.dense.bias)
        return out
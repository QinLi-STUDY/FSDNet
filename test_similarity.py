import torch

# 假设 A 和 B 是你的两个 Tensor，维度均为 [16, 256, 64, 64]
A = torch.randn(16, 256, 64, 64)
B = torch.randn(16, 256, 64, 64)

# 计算范数
A_norm = A / A.norm(dim=(-1, -2), keepdim=True)
B_norm = B / B.norm(dim=(-1, -2), keepdim=True)

# 计算余弦相似度
similarity = (A_norm * B_norm).sum(dim=(-1, -2))

# similarity 的维度为 [16, 256]
# 如果需要将其扩展到 [16, 256, 64, 64]，可以使用 unsqueeze 和 expand
similarity_expanded = similarity.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 64, 64)
print(similarity_expanded.shape)
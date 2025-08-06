import torch  
import torch.nn.functional as F  
  
  
# 示例使用  
batch_size = 32  
embedding_dim = 128  
  
# 随机生成嵌入向量作为示例  
image_embedding = torch.randn(batch_size, embedding_dim)  
text_embedding = torch.randn(batch_size, embedding_dim)  
  
# 计算余弦相似度损失  
loss = similarity_wenben_loss = torch.mean(1 - (image_embedding * text_embedding).sum(dim=1)) 
print(loss)

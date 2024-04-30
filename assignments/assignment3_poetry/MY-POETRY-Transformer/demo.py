import torch
from dataset import get_datasets
from model import make_model, subsequent_mask
from opt import LabelSmoothing, NoamOpt
from utils import AverageMeter
from torch.utils.data import  DataLoader
import tqdm
import numpy as np
from torch.autograd import Variable


def predict(model, src, src_mask, tgt, max_len, end_symbol, method='greedy', temperature=1.0):
    """
    使用给定模型进行预测。

    参数:
        model: 使用的神经网络模型。
        src: 输入序列的张量。
        src_mask: 输入序列的掩码张量。
        tgt: 目标序列的张量。
        max_len: 最大生成序列的长度。
        start_symbol: 开始符号索引。
        end_symbol: 结束符号索引。
        method: 生成策略，可以是'greedy'（贪婪搜索）或'sample'（采样）。
        temperature: 用于控制采样的温度参数。

    返回:
        生成的目标序列张量。
    """
    memory = model.encode(src, src_mask)
    tgt = tgt.to(device)
    for i in range(max_len-1):
      pad = word2ix['<pad>']
      tgt_mask1 = (tgt != pad).unsqueeze(-2)
      tgt_mask2 = torch.Tensor(subsequent_mask(tgt.size(-1)).type_as(tgt_mask1.data)).to(device)
      tgt_mask = tgt_mask1 & tgt_mask2 # 1,t,t
      tgt_mask = subsequent_mask(tgt.size(1)).type_as(src).to(device)
      out = model.decode(memory, src_mask, 
                          tgt,
                          tgt_mask).to(device)
      log_prob = model.generator(out[:, -1])
      prob = log_prob.exp()
      if method == 'greedy':
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        
      elif method == 'sample':
        prob = prob**(1.0/temperature)
        prob = prob/prob.sum(-1, keepdim=True)
        import torch.nn.functional as F
        next_word = torch.multinomial(prob, 1).squeeze().detach().item()
        
      tgt = torch.cat([tgt, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
      if next_word == end_symbol:
        return tgt
    return tgt


def generate_poetry(title="夏日繁花", content=""):
  """
  根据标题和正文内容，生成完整的正文

  参数:
      title: 标题
      content: 正文

  返回:
      完整的正文
  """
  title_index = [word2ix['<bos>']] + [word2ix[c] for c in list(title)] + [word2ix['<eos>']]
  content_index = [word2ix['<bos>']]  + [word2ix[c] for c in list(content)]
  src = Variable(torch.LongTensor([title_index])).to(device)
  tgt = torch.Tensor(np.array(content_index)).long().view(1,len(content_index))
  
  pad = word2ix['<pad>']
  src_mask = (src != pad).unsqueeze(-2).to(device)
  
  predict_idx = predict(model, src, src_mask, tgt, max_len=26, end_symbol=word2ix['<eos>'], method='greedy')
  predict_str = "".join([ix2word[c] for c in predict_idx[0].cpu().numpy()])
  print(f"输入的标题: {title}")
  print(f"输入的正文: {content}")
  print(f"预测的正文: {predict_str}")
  return predict_str
  


if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  train_dataset, val_dataset, ix2word, word2ix = get_datasets()
  V=len(ix2word)
  model = make_model(V, V, N=3, d_model=512, d_ff=2048, h=8, dropout=0).to(device) # 模型只有两层
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # 加载模型文件 
  save_dict = torch.load("model_latest_3layer_24word.pt")
  model.load_state_dict(save_dict['state_dict'])
  print(f"==loaded state dict from epoch {save_dict['epoch']}")
  model.eval()
  
  content = generate_poetry(title="夏日繁花")
  content = generate_poetry(title="夏日繁花", content="夏日校园中，")

  
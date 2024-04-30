import json
import zhconv
import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import glob
  

def get_datasets():
  Paragraph_Max = 24
  Title_Max = 12
  file_list = glob.glob('chinese-poetry/全唐诗/poet.song.*.json')
  raw_data = []
  filepath = sorted(file_list)
  for filepath in file_list:
    
    raw_data += json.load(open(filepath))
  
  data = convert_to_simplified(raw_data)
  data = filter_data(data, Paragraph_Max, Title_Max)
  
  ix2word, word2ix = create_vocabulary(data)
  
  data = convert_to_indexes(data, word2ix, Paragraph_Max, Title_Max)
  print(f"=== total data: {len(data)}")
  print(f"=== total vocab: {len(ix2word)}")
  
  train_size = int(len(data)*0.8)

  train_dataset = PoetryDataset(data[:train_size], word2ix)
  val_dataset = PoetryDataset(data[train_size:], word2ix)
  return train_dataset, val_dataset, ix2word, word2ix

  
  
# 转化为简体中文
def convert_to_simplified(data):
    """Converts data containing author, title, and paragraphs to Simplified Chinese.

    Args:
        data (list): A list of dictionaries, where each dictionary represents a piece
                     of content with keys 'author', 'title', and 'paragraphs'. The
                     paragraphs are expected to be a list of strings.

    Returns:
        list: A new list of dictionaries with the same structure as the input data,
              but with the author, title, and paragraphs converted to Simplified Chinese.
    """

    simplified_data = []
    for item in data:
        new_item = {
            'author': zhconv.convert(item['author'], 'zh-cn'),
            'title': zhconv.convert(item['title'], 'zh-cn'),
            'paragraphs': ''.join([zhconv.convert(paragraph, 'zh-cn') for paragraph in item['paragraphs']])
        }
        simplified_data.append(new_item)
    return simplified_data



# 过滤数据
def filter_data(data, Paragraph_Max, Title_Max):
    # data = [c for c in data if len(c['paragraphs'])> 0 and len(c['paragraphs'])<= Paragraph_Max]
    data = [c for c in data if len(c['paragraphs']) == 24]
    data = [c for c in data if len(c['title'])> 0 and len(c['title'])<= Title_Max]
    return data



# 创建字典
def create_vocabulary(data):
    vocabulary = set()
    for item in data:
        vocabulary.update(list(item['title']))
        vocabulary.update(list(item['paragraphs']))
    ix2word = sorted(list(vocabulary)) + ['<bos>','<eos>','<sep>','<pad>']
    word2ix = {v:k for k,v in enumerate(ix2word)}
    return ix2word, word2ix



# 将title和paragraphs转化为token index
def convert_to_indexes(data, word2ix, Paragraph_Max, Title_Max):
    """Converts title and paragraphs in data to sequences of token indices with padding.

    Args:
        data (list): A list of dictionaries with 'title' and 'paragraphs' keys.
        word2ix (dict): A dictionary mapping words to their indices.
        pad_idx (int, optional): Index of the padding token. Defaults to '<pad>'.

    Returns:
        list: A new list of dictionaries with the same structure as the input data,
            but with 'title' and 'paragraphs' converted to padded lists of token indices.
    """
    pad_idx = word2ix['<pad>']
    processed_data = []
    for item in data:
        title_index = [word2ix[c] for c in list(item['title'])]
        if len(title_index) <= Title_Max:
          title_index = [word2ix['<bos>']] + title_index + [word2ix['<eos>']]+[pad_idx] * (Title_Max - len(item['title']))
        else:
          title_index = [word2ix['<bos>']]+ title_index[:Title_Max] + [word2ix['<eos>']]
        
        paragraph_index = [word2ix[c] for c in list(item['paragraphs'])]   
        if len(item['paragraphs']) <= Paragraph_Max:
          paragraph_index = [word2ix['<bos>']] + paragraph_index + [word2ix['<eos>']] + [pad_idx] * (Paragraph_Max - len(item['paragraphs']))
        else:  
          paragraph_index = [word2ix['<bos>']] + paragraph_index[:Paragraph_Max] + [word2ix['<eos>']]
        
        
        title_paragraph_index = [word2ix[c] for c in list(item['title'])] + [word2ix['<pad>']] + [word2ix[c] for c in list(item['paragraphs'])]   
        title_paragraph_Max = Title_Max + Paragraph_Max + 1
        if len(title_paragraph_index) <= title_paragraph_Max:
          title_paragraph_index = [word2ix['<bos>']] + title_paragraph_index + [word2ix['<eos>']] + [pad_idx] * (title_paragraph_Max - len(title_paragraph_index))
        else:  
          title_paragraph_index = [word2ix['<bos>']] + paragraph_index[:title_paragraph_Max] + [word2ix['<eos>']]
        
        new_item = {
            'author': item['author'],
            'title': title_index,
            'paragraphs': paragraph_index,
            'title_paragraphs': title_paragraph_index
        }
        processed_data.append(new_item)
    return processed_data




class PoetryDataset(Dataset):
  """Custom Dataset class for poems."""

  def __init__(self, data, word2ix):
    """
    Args:
      data (list): List of dictionaries containing processed poems.
      word2ix (dict): Dictionary mapping words to their indices.
    """
    self.data = data
    self.word2ix = word2ix

  def __len__(self):
    """Returns the total number of poems in the dataset."""
    return len(self.data)

  def __getitem__(self, idx):
    """Returns a single poem (title and paragraphs) as tensors.

    Args:
      idx (int): Index of the poem in the dataset.

    Returns:
      tuple: A tuple containing two tensors:
        - title_tensor (torch.Tensor): Title as a sequence of token indices.
        - paragraph_tensor (torch.Tensor): Paragraphs as a sequence of token indices.
    """
    poem = self.data[idx]
    title_tensor = torch.tensor(poem['title'], dtype=torch.long)
    paragraph_tensor = torch.tensor(poem['paragraphs'], dtype=torch.long)
    return title_tensor, paragraph_tensor


class PoetryDataset(Dataset):
  """Custom Dataset class for poems."""

  def __init__(self, data, word2ix):
    """
    Args:
      data (list): List of dictionaries containing processed poems.
      word2ix (dict): Dictionary mapping words to their indices.
    """
    self.data = data
    self.word2ix = word2ix

  def __len__(self):
    """Returns the total number of poems in the dataset."""
    return len(self.data)

  def __getitem__(self, idx):
    """Returns a single poem (title and paragraphs) as tensors.

    Args:
      idx (int): Index of the poem in the dataset.

    Returns:
      tuple: A tuple containing two tensors:
        - title_tensor (torch.Tensor): Title as a sequence of token indices.
        - paragraph_tensor (torch.Tensor): Paragraphs as a sequence of token indices.
    """
    poem = self.data[idx]
    # title_tensor = torch.tensor(poem['title'], dtype=torch.long)
    # paragraph_tensor = torch.tensor(poem['paragraphs'], dtype=torch.long)
    title_paragraph_tensor = torch.tensor(poem['title_paragraphs'], dtype=torch.long)
    return title_paragraph_tensor

import torch
from dataset import get_datasets
from model import make_model, subsequent_mask
from opt import LabelSmoothing, NoamOpt
from utils import AverageMeter
from torch.utils.data import  DataLoader
import tqdm
import numpy as np
import matplotlib.pyplot as plt

def val_epoch(dataloader, model, epoch):
    model.eval()
    lossMeter = AverageMeter()
    accMeter = AverageMeter()
    n = dataloader.size  if hasattr(dataloader,'size') else len(dataloader) 
    loop = tqdm.tqdm(dataloader,total=n)
    loop.set_description(f"Val Epoch {epoch}")
    step = 0
    for x, y in loop:
        step += 1
        x = x.to(device)
        y = y.to(device)
        src = x # b,s
        src_mask = (src != pad).unsqueeze(-2) # b,1,s
        
        tgt = y[:,:-1] # target input: b,t
        tgt_gt = y[:,1:] # target ground truth outpput: b,t
        tgt_mask = (tgt != pad).unsqueeze(-2) #add query dimension(for expand) b,1,t
        tgt_mask = tgt_mask & torch.Tensor(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)) # 1,t,t
        with torch.no_grad():
          out = model.forward(src, tgt, src_mask, tgt_mask) # b,l,d
          log_prob = model.generator(out) # log of probabiliy: b,l,v
          loss = criterion(log_prob.contiguous().view(-1, log_prob.size(-1)), 
                                    tgt_gt.contiguous().view(-1)) 
        
        
        lossMeter.update(loss.item(), num=x.size(0))
        predictions = torch.argmax(log_prob.contiguous().view(-1, log_prob.size(-1)), dim=-1)
        accs = predictions == tgt_gt.contiguous().view(-1)
        valid_token = tgt_gt.contiguous().view(-1)!=pad
        acc = ((accs*valid_token).sum()/max(valid_token.sum(),1e-5)).item()
        accMeter.update(acc, valid_token.sum().item())
        loop.set_postfix({"avg loss":lossMeter.avg, "avg acc": accMeter.avg})
    return lossMeter.avg, accMeter.avg


def train_epoch(train_dataloader,  model, opt, epoch=1):
    model.train()
    lossMeter = AverageMeter()
    accMeter = AverageMeter()
    n = train_dataloader.size  if hasattr(train_dataloader,'size') else len(train_dataloader) 
    loop = tqdm.tqdm(train_dataloader, total=n)
    loop.set_description(f"Train Epoch {epoch}")
    step = 0
    for x, y in loop:
        step+=1
        x = x.to(device)
        y = y.to(device)
        
        src = x # b,s
        src_mask = (src != pad).unsqueeze(-2) # b,1,s
        
        tgt = y[:,:-1] # target input: b,t
        tgt_gt = y[:,1:] # target ground truth outpput: b,t
        tgt_mask = (tgt != pad).unsqueeze(-2) #add query dimension(for expand) b,1,t
        tgt_mask = tgt_mask & torch.Tensor(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)) # 1,t,t
        
        # Todo: 调用模型产生output
        # out = ... (code here)
        
        log_prob = model.generator(out) # log of probabiliy: b,l,v
        loss = criterion(log_prob.contiguous().view(-1, log_prob.size(-1)), 
                                  tgt_gt.contiguous().view(-1)) 
        
        # Todo: 增加loss的梯度反向传播调用
        # (code here)
        
        opt.step()
        opt.optimizer.zero_grad()
        lossMeter.update(loss.item(), num=x.size(0))
        if epoch == 20:
          print("epoch 20")
        predictions = torch.argmax(log_prob.contiguous().view(-1, log_prob.size(-1)), dim=-1)
        accs = predictions == tgt_gt.contiguous().view(-1)
        valid_token = tgt_gt.contiguous().view(-1)!=pad
        acc = (((accs*valid_token).sum()/max(valid_token.sum(),1e-5))).item()
        accMeter.update(acc, valid_token.sum().item())
        loop.set_postfix({"avg loss":lossMeter.avg, "avg acc": accMeter.avg})
    return lossMeter.avg,  accMeter.avg

def train(train_dataloader, val_dataloader, model, opt, num_epochs, exp_name=''):
    metrics = []
    best_val_loss = np.inf
    best_acc = 0
    best_epoch = -1
    for epoch in range(1,num_epochs+1):
      train_loss, train_acc = train_epoch(train_dataloader, model, opt, epoch=epoch)
      val_loss, val_acc = val_epoch(val_dataloader, model,epoch)
      metrics.append({
          "epoch": epoch,
          "train_loss": train_loss,
          "train_acc": train_acc,
          "val_loss": val_loss,
          "val_acc": val_acc,
      })
      # if val_loss < best_val_loss:
      if val_acc > best_acc:
          best_acc = val_acc
          best_val_loss = val_loss
          best_epoch = epoch
          save_dict = {
            "state_dict": model.state_dict(),
            "epoch": epoch
          }
          torch.save(save_dict, f"model_best_{exp_name}.pt")
          
  
      # Plotting the metrics
      epochs = [metric["epoch"] for metric in metrics]
      train_losses = [metric["train_loss"] for metric in metrics]
      val_losses = [metric["val_loss"] for metric in metrics]
      train_accs = [metric["train_acc"] for metric in metrics]
      val_accs = [metric["val_acc"] for metric in metrics]
              
      plt.figure(figsize=(10, 5))
      plt.subplot(1, 2, 1)
      plt.plot(epochs, train_losses, label='Train Loss')
      plt.plot(epochs, val_losses, label='Val Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend()
      

      plt.subplot(1, 2, 2)
      plt.plot(epochs, train_accs, label='Train Accuracy')
      plt.plot(epochs, val_accs, label='Val Accuracy')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.legend()
      plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Val Acc Epoch ({best_epoch})')

      plt.tight_layout()

      # Save the plot as a PNG
      plt.savefig(f'metrics_{exp_name}_plot.png')
    
    # end of training  
    save_dict = {
            "state_dict": model.state_dict(),
            "epoch": epoch
          }
    torch.save(save_dict, f"model_latest_{exp_name}.pt")
    print(f"==best_epoch:{best_epoch}")
    return metrics
    

if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  train_dataset, val_dataset, ix2word, word2ix = get_datasets()
  # Create the dataloader
  batch_size = 64  # Number of poems per batch
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  V=len(ix2word)
  model = make_model(V, V, N=3, d_model=512, d_ff=2048, h=8, dropout=0.0).to(device) # 模型只有两层
  pad = word2ix['<pad>']
  criterion = LabelSmoothing(size=V, padding_idx=pad, smoothing=0.2)
  opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
          torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
  metrics = train(train_dataloader, val_dataloader, model, opt, num_epochs=40, exp_name='3layer_24word')
  
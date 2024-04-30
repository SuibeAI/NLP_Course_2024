class AverageMeter:
  def __init__(self):
    self.reset()

  def reset(self):
    self.count = 0
    self.sum = 0.0
    self.val = 0.0
    self.avg = 0.0

  def update(self, val, num=1):
    self.val = val
    self.sum += val * num
    self.count += num
    self.avg = self.sum / (self.count if self.count > 0 else 1)  # Avoid division by zero


class MovingAverageMeter:
  def __init__(self, momentum=0.9):
    """
    Args:
      momentum (float, optional): Momentum for smoothing the average. Defaults to 0.9.
    """
    self.momentum = momentum
    self.current_loss = None
    self.moving_avg_loss = 0.0

  def update(self, loss):
    if self.current_loss is None:
      self.current_loss = loss
    else:
      self.current_loss = self.moving_avg_loss * self.momentum + loss * (1 - self.momentum)
    self.moving_avg_loss = self.current_loss

  def get(self):
    return self.moving_avg_loss


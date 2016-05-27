import matplotlib.pyplot as plt

def plot_from_file(filename):
  train_loss = []
  val_loss = []
  train_acc = []
  val_acc = []

  for line in open(filename):
    line_arr = line.strip().split(',')
    if line_arr[0] == 'train_loss':
      train_loss.append(float(line_arr[1]))
    if line_arr[0] == 'val_loss':
      val_loss.append(float(line_arr[1]))
    if line_arr[0] == 'train_acc':
      train_acc.append(float(line_arr[1]))
    if line_arr[0] == 'val_acc':
      val_acc.append(float(line_arr[1]))

  train_plot, = plt.plot(xrange(len(train_loss)), train_loss, label='Train loss')
  val_plot, = plt.plot(xrange(len(val_loss)), val_loss, label = 'Val loss')
  plt.title('Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss value')
  plt.legend(handles=[train_plot, val_plot])
  plt.show()

  train_acc_plot, = plt.plot(xrange(len(train_acc)), train_acc, label='Train accuracy')
  val_acc_plot, = plt.plot(xrange(len(val_acc)), val_acc, label = 'Val accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy value')
  plt.legend(handles=[train_acc_plot, val_acc_plot])
  plt.show()

plot_from_file('outputs/dmn/lr_0.001_hs_50_e_100.txt')
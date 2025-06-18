from optimization_methods import *

# test learning rates
epochs = [1,2,3,4,50,100,200,300,500,700,1000]
lr_0 = 0.001
decay_rate = 1
time_interval = 100

for i, epoch in enumerate(epochs):
  print("Learning rate after epoch %i: %f" % (epochs[i], schedule_lr_decay(
                    lr_0, epoch, decay_rate, time_interval)))

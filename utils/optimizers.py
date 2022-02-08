from tflearn.optimizers import RMSProp, Adam, AdaGrad, Momentum

# Optimizer 
rmsprop = RMSProp(learning_rate=0.1, decay=0.999)
adam = Adam(learning_rate=0.001, beta1=0.99)
adagrad = AdaGrad(learning_rate=0.01, initial_accumulator_value=0.01)
momentum = Momentum(learning_rate=0.01, lr_decay=0.96, decay_step=100)

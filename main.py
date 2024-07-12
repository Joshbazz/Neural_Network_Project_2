from LSTMModel import LSTMModel
from OptunaOptimizer import HyperParameterOptimizer

# instantiate the model, and run it
model = LSTMModel(test_size=0.25,
                  lstm_neurons = 50,
                  dropout_rate=0.50,
                  dense_neurons=10, 
                  learning_rate=0.0001, 
                  epochs=50, 
                  batch_size=32, 
                  validation_split=0.25, 
                  plot=True)

model.run_and_train()


# run the optimizer
optimizer = HyperParameterOptimizer(model.data)
optimizer.optimize(n_trials=5, timeout=500)
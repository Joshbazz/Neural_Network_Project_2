from LSTMModel import LSTMModel
from OptunaOptimizer import HyperParameterOptimizer

# instantiate the model, and run it
model = LSTMModel(model_path=None,
                  #data_path='fear_greed_btc_combined.csv',
                  test_size=0.25,
                  lstm_neurons = 200,
                  dropout_rate=0.50,
                  dense_neurons=4, 
                  learning_rate=0.001, 
                  epochs=50, 
                  batch_size=16, 
                  validation_split=0.25, 
                  plot=True)

model.run_and_train()
# model.run_with_pretrained()

# run the optimizer
optimizer = HyperParameterOptimizer(model.data, 
                                    batch_size=model.batch_size, 
                                    validation_split=model.validation_split, 
                                    epochs=model.epochs)

optimizer.optimize(n_trials=5, timeout=500)
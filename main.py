from LSTMModel import LSTMModel
from OptunaOptimizer import HyperParameterOptimizer

# instantiate the model, and run it
model = LSTMModel(model_path=None,
                  #data_path='fear_greed_btc_combined.csv',
                  test_size=0.25,
                  lstm_neurons = 88,
                  dropout_rate=0.06193765645303727,
                  dense_neurons=35, 
                  learning_rate=8.676313161215016e-05, 
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
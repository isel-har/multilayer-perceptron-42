import json

hyper_params = {
    "learning_rate":[0.01, 0.001],
    "batch_size":[16, 32],
    "epochs" : [20, 40, 80, 100],
    "optimizer":["gd", "adam"],
    "layers": [
        {
          "size": 32,
          "activation": "relu"
        },
        {
          "size": 2,
          "activation": "softmax"
        }
      ]
}



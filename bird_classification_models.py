import hpo
import tensorflow as tf
import numpy as np


optimiser = hpo.Optimiser(optimiser_name="optimiser_adam", optimiser_type=tf.keras.optimizers.Adam, hyperparameters=[
    hpo.Parameter(parameter_name="learning_rate", parameter_value=0.0001, value_range=[1 * (10 ** n) for n in range(0, -7, -1)]) #np.arange(0.0001, 1.0, 0.0005).tolist())
])

cnn = [
    hpo.Layer(layer_name="input_layer_conv_2d", layer_type=tf.keras.layers.Conv2D,
    parameters=[
        hpo.Parameter(parameter_name="padding", parameter_value="same", value_range=["same", "valid"], constraints=None),#need to add more
        hpo.Parameter(parameter_name="input_shape", parameter_value=(224, 224, 3))
    ],
    hyperparameters=[
        hpo.Parameter(parameter_name="filters", parameter_value=256, value_range=[2**x for x in range(4, 9)], constraints=None),# range from 16-512
        hpo.Parameter(parameter_name="kernel_size", parameter_value=5, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None, encode_string_values=True)
    ]),

    hpo.Layer(layer_name="hidden_layer_1_dropout", layer_type=tf.keras.layers.Dropout,
    parameters=[
      # hpo.Parameter(parameter_name="seed", parameter_value=42)
    ],
    hyperparameters=[
      hpo.Parameter(parameter_name="rate", parameter_value=0.1,
                    value_range=np.arange(0.0, 0.4, 0.05).tolist(), constraints=None)
    ]),

    hpo.Layer(layer_name="hidden_layer_2_max_pooling", layer_type=tf.keras.layers.MaxPooling2D,
    parameters=[
        hpo.Parameter(parameter_name="pool_size", parameter_value=(3, 3))
    ],
    hyperparameters=[]),

    hpo.Layer(layer_name="hidden_layer_3_conv_2d", layer_type=tf.keras.layers.Conv2D,
    parameters=[
        hpo.Parameter(parameter_name="padding", parameter_value="same", value_range=["same", "valid"], constraints=None)#need to add more
    ],
    hyperparameters=[
        hpo.Parameter(parameter_name="filters", parameter_value=256, value_range=[2**x for x in range(4, 9)], constraints=None),# range from 16-512
        hpo.Parameter(parameter_name="kernel_size", parameter_value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None, encode_string_values=True)
    ]),

    hpo.Layer(layer_name="hidden_layer_4_max_pooling", layer_type=tf.keras.layers.MaxPooling2D,
    parameters=[
        hpo.Parameter(parameter_name="pool_size", parameter_value=(3, 3))
    ],
    hyperparameters=[]),

    #4 - Conv 2D - Optimise
    hpo.Layer(layer_name="hidden_layer_5_conv_2d", layer_type=tf.keras.layers.Conv2D,
    parameters=[
        hpo.Parameter(parameter_name="padding", parameter_value="same", value_range=["same", "valid"], constraints=None)#need to add more
    ],
    hyperparameters=[
        hpo.Parameter(parameter_name="filters", parameter_value=512, value_range=[2**x for x in range(4, 10)], constraints=None),# range from 16-512
        hpo.Parameter(parameter_name="kernel_size", parameter_value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None, encode_string_values=True)
    ]),

    hpo.Layer(layer_name="hidden_layer_6_dropout", layer_type=tf.keras.layers.Dropout,
    parameters=[
      # hpo.Parameter(parameter_name="seed", parameter_value=42)
    ],
    hyperparameters=[
      hpo.Parameter(parameter_name="rate", parameter_value=0.1,
                    value_range=np.arange(0.0, 0.4, 0.05).tolist(), constraints=None)
    ]),

    hpo.Layer(layer_name="hidden_layer_7_max_pooling", layer_type=tf.keras.layers.MaxPooling2D,
    parameters=[
        hpo.Parameter(parameter_name="pool_size", parameter_value=(3, 3))
    ],
    hyperparameters=[]),

    #6 - Conv 2D - Optimise
    hpo.Layer(layer_name="hidden_layer_8_conv_2d", layer_type=tf.keras.layers.Conv2D,
    parameters=[
        hpo.Parameter(parameter_name="padding", parameter_value="same", value_range=["same", "valid"], constraints=None)#need to add more
    ],
    hyperparameters=[
        hpo.Parameter(parameter_name="filters", parameter_value=512, value_range=[2**x for x in range(4, 10)], constraints=None),# range from 16-512
        hpo.Parameter(parameter_name="kernel_size", parameter_value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None, encode_string_values=True)#need to add more
    ]),

    hpo.Layer(layer_name="hidden_layer_9_max_pooling", layer_type=tf.keras.layers.MaxPooling2D,
    parameters=[
        hpo.Parameter(parameter_name="pool_size", parameter_value=(3, 3))
    ],
    hyperparameters=[]),

    hpo.Layer(layer_name="hidden_layer_10_flatten", layer_type=tf.keras.layers.Flatten,
    parameters=[],
    hyperparameters=[]),

    hpo.Layer(layer_name="hidden_layer_11_dropout", layer_type=tf.keras.layers.Dropout,
    parameters=[
        hpo.Parameter(parameter_name="seed", parameter_value=42)
    ],
    hyperparameters=[
        hpo.Parameter(parameter_name="rate", parameter_value=0.2, value_range=np.arange(0.0, 0.4, 0.05).tolist(), constraints=None)
    ]),

    hpo.Layer(layer_name="hidden_layer_12_dense", layer_type=tf.keras.layers.Dense,
    parameters=[
    ],
    hyperparameters=[
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["tanh", "sigmoid", "relu"], constraints=None, encode_string_values=True),#need to add more
        hpo.Parameter(parameter_name="units", parameter_value=1024, value_range=[2**x for x in range(7, 12)], constraints=None)#range between 4 and 4096
    ]),

    hpo.Layer(layer_name="hidden_layer_13_dropout", layer_type=tf.keras.layers.Dropout,
    parameters=[
        #hpo.Parameter(parameter_name="seed", parameter_value=42)
    ],
    hyperparameters=[
        hpo.Parameter(parameter_name="rate", parameter_value=0.15, value_range=np.arange(0.0, 0.4, 0.05).tolist(), constraints=None)
    ]),

    hpo.Layer(layer_name="hidden_layer_14_dense", layer_type=tf.keras.layers.Dense,
    parameters=[
    ],
    hyperparameters=[
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["tanh", "sigmoid", "relu"], constraints=None, encode_string_values=True),#need to add more
        hpo.Parameter(parameter_name="units", parameter_value=512, value_range=[2**x for x in range(6, 11)], constraints=None)#range between 4 and 4096
    ]),

    hpo.Layer(layer_name="hidden_layer_15_dropout", layer_type=tf.keras.layers.Dropout,
    parameters=[
        #hpo.Parameter(parameter_name="seed", parameter_value=42)
    ],
    hyperparameters=[
        hpo.Parameter(parameter_name="rate", parameter_value=0.1, value_range=np.arange(0.0, 0.4, 0.05).tolist(), constraints=None)
    ]),

    hpo.Layer(layer_name="output_layer_dense", layer_type=tf.keras.layers.Dense,
    parameters=[
        hpo.Parameter(parameter_name="units", parameter_value=150)
    ],
    hyperparameters=[
        hpo.Parameter(parameter_name="activation", parameter_value="sigmoid", value_range=["tanh", "sigmoid", "softmax"], constraints=None, encode_string_values=True)#need to add more
    ])]

# import hpo
# import tensorflow as tf
# import numpy as np
#
#
# _optimiser = hpo.Optimiser(optimiser_name="optimiser_adam", optimiser_type=tf.keras.optimizers.Adam, hyperparameters=[
#     hpo.Parameter(parameter_name="learning_rate", parameter_value=0.0001, value_range=[1 * (10 ** n) for n in range(0, -7, -1)]) #np.arange(0.0001, 1.0, 0.0005).tolist())
# ])
#
# cnn = [
#     hpo.Layer(layer_name="input_layer_conv_2d", layer_type=tf.keras.layers.Conv2D,
#     parameters=[
#         hpo.Parameter(parameter_name="padding", parameter_value="same", value_range=["same", "valid"], constraints=None),#need to add more
#         hpo.Parameter(parameter_name="input_shape", parameter_value=(224, 224, 3))
#     ],
#     hyperparameters=[
#         hpo.Parameter(parameter_name="filters", parameter_value=64, value_range=[2**x for x in range(4, 10)], constraints=None),# range from 16-512
#         hpo.Parameter(parameter_name="kernel_size", parameter_value=5, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
#         hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None, encode_string_values=True)
#     ]),
#
#     hpo.Layer(layer_name="hidden_layer_1_dropout", layer_type=tf.keras.layers.Dropout,
#     parameters=[
#       # hpo.Parameter(parameter_name="seed", parameter_value=42)
#     ],
#     hyperparameters=[
#       hpo.Parameter(parameter_name="rate", parameter_value=0.1,
#                     value_range=np.arange(0.0, 0.4, 0.05).tolist(), constraints=None)
#     ]),
#
#     hpo.Layer(layer_name="hidden_layer_2_max_pooling", layer_type=tf.keras.layers.MaxPooling2D,
#     parameters=[
#         hpo.Parameter(parameter_name="pool_size", parameter_value=(3, 3))
#     ],
#     hyperparameters=[]),
#
#     hpo.Layer(layer_name="hidden_layer_3_conv_2d", layer_type=tf.keras.layers.Conv2D,
#     parameters=[
#         hpo.Parameter(parameter_name="padding", parameter_value="same", value_range=["same", "valid"], constraints=None)#need to add more
#     ],
#     hyperparameters=[
#         hpo.Parameter(parameter_name="filters", parameter_value=128, value_range=[2**x for x in range(4, 10)], constraints=None),# range from 16-512
#         hpo.Parameter(parameter_name="kernel_size", parameter_value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
#         hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None, encode_string_values=True)
#     ]),
#
#     hpo.Layer(layer_name="hidden_layer_4_max_pooling", layer_type=tf.keras.layers.MaxPooling2D,
#     parameters=[
#         hpo.Parameter(parameter_name="pool_size", parameter_value=(3, 3))
#     ],
#     hyperparameters=[]),
#
#     #4 - Conv 2D - Optimise
#     hpo.Layer(layer_name="hidden_layer_5_conv_2d", layer_type=tf.keras.layers.Conv2D,
#     parameters=[
#         hpo.Parameter(parameter_name="padding", parameter_value="same", value_range=["same", "valid"], constraints=None)#need to add more
#     ],
#     hyperparameters=[
#         hpo.Parameter(parameter_name="filters", parameter_value=256, value_range=[2**x for x in range(4, 10)], constraints=None),# range from 16-512
#         hpo.Parameter(parameter_name="kernel_size", parameter_value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
#         hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None, encode_string_values=True)
#     ]),
#
#     hpo.Layer(layer_name="hidden_layer_6_dropout", layer_type=tf.keras.layers.Dropout,
#     parameters=[
#       # hpo.Parameter(parameter_name="seed", parameter_value=42)
#     ],
#     hyperparameters=[
#       hpo.Parameter(parameter_name="rate", parameter_value=0.1,
#                     value_range=np.arange(0.0, 0.4, 0.05).tolist(), constraints=None)
#     ]),
#
#     hpo.Layer(layer_name="hidden_layer_7_max_pooling", layer_type=tf.keras.layers.MaxPooling2D,
#     parameters=[
#         hpo.Parameter(parameter_name="pool_size", parameter_value=(3, 3))
#     ],
#     hyperparameters=[]),
#
#     #6 - Conv 2D - Optimise
#     hpo.Layer(layer_name="hidden_layer_8_conv_2d", layer_type=tf.keras.layers.Conv2D,
#     parameters=[
#         hpo.Parameter(parameter_name="padding", parameter_value="same", value_range=["same", "valid"], constraints=None)#need to add more
#     ],
#     hyperparameters=[
#         hpo.Parameter(parameter_name="filters", parameter_value=512, value_range=[2**x for x in range(4, 10)], constraints=None),# range from 16-512
#         hpo.Parameter(parameter_name="kernel_size", parameter_value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
#         hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None, encode_string_values=True)#need to add more
#     ]),
#
#     hpo.Layer(layer_name="hidden_layer_9_max_pooling", layer_type=tf.keras.layers.MaxPooling2D,
#     parameters=[
#         hpo.Parameter(parameter_name="pool_size", parameter_value=(3, 3))
#     ],
#     hyperparameters=[]),
#
#     hpo.Layer(layer_name="hidden_layer_10_flatten", layer_type=tf.keras.layers.Flatten,
#     parameters=[],
#     hyperparameters=[]),
#
#     hpo.Layer(layer_name="hidden_layer_11_dropout", layer_type=tf.keras.layers.Dropout,
#     parameters=[
#         hpo.Parameter(parameter_name="seed", parameter_value=42)
#     ],
#     hyperparameters=[
#         hpo.Parameter(parameter_name="rate", parameter_value=0.2, value_range=np.arange(0.0, 0.4, 0.05).tolist(), constraints=None)
#     ]),
#
#     hpo.Layer(layer_name="hidden_layer_12_dense", layer_type=tf.keras.layers.Dense,
#     parameters=[
#     ],
#     hyperparameters=[
#         hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["tanh", "sigmoid", "relu"], constraints=None, encode_string_values=True),#need to add more
#         hpo.Parameter(parameter_name="units", parameter_value=1024, value_range=[2**x for x in range(7, 13)], constraints=None)#range between 4 and 4096
#     ]),
#
#     hpo.Layer(layer_name="hidden_layer_13_dropout", layer_type=tf.keras.layers.Dropout,
#     parameters=[
#         #hpo.Parameter(parameter_name="seed", parameter_value=42)
#     ],
#     hyperparameters=[
#         hpo.Parameter(parameter_name="rate", parameter_value=0.15, value_range=np.arange(0.0, 0.4, 0.05).tolist(), constraints=None)
#     ]),
#
#     hpo.Layer(layer_name="hidden_layer_14_dense", layer_type=tf.keras.layers.Dense,
#     parameters=[
#     ],
#     hyperparameters=[
#         hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["tanh", "sigmoid", "relu"], constraints=None, encode_string_values=True),#need to add more
#         hpo.Parameter(parameter_name="units", parameter_value=512, value_range=[2**x for x in range(6, 13)], constraints=None)#range between 4 and 4096
#     ]),
#
#     hpo.Layer(layer_name="hidden_layer_15_dropout", layer_type=tf.keras.layers.Dropout,
#     parameters=[
#         #hpo.Parameter(parameter_name="seed", parameter_value=42)
#     ],
#     hyperparameters=[
#         hpo.Parameter(parameter_name="rate", parameter_value=0.1, value_range=np.arange(0.0, 0.4, 0.05).tolist(), constraints=None)
#     ]),
#
#     hpo.Layer(layer_name="output_layer_dense", layer_type=tf.keras.layers.Dense,
#     parameters=[
#         hpo.Parameter(parameter_name="units", parameter_value=150)
#     ],
#     hyperparameters=[
#         hpo.Parameter(parameter_name="activation", parameter_value="sigmoid", value_range=["tanh", "sigmoid", "softmax"], constraints=None, encode_string_values=True)#need to add more
#     ])]
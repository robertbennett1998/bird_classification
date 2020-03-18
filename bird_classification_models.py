import hpo
import tensorflow as tf

# https://towardsdatascience.com/vgg-neural-networks-the-next-step-after-alexnet-3f91fa9ffe2c
VGG_A_11 = [
    hpo.Layer(layer_name="input_layer_conv_2d", layer_type=tf.keras.layers.Conv2D,
        parameters=[
            hpo.Parameter(parameter_name="input_shape", parameter_value=(224, 224, 3)),
            hpo.Parameter(parameter_name="kernel_size", parameter_value=(3, 3)),
            hpo.Parameter(parameter_name="filters", parameter_value=64),
        ],
        hyperparameters=[
        ]),

    hpo.Layer(layer_name="hidden_layer_max_pooling_2d_1", layer_type=tf.keras.layers.MaxPooling2D,
        parameters=[
        ],
        hyperparameters=[
        ]),

    hpo.Layer(layer_name="hidden_layer_conv_2d_1", layer_type=tf.keras.layers.Conv2D,
        parameters=[
          hpo.Parameter(parameter_name="kernel_size", parameter_value=(3, 3)),
          hpo.Parameter(parameter_name="filters", parameter_value=128)
        ],
        hyperparameters=[
        ]),

    hpo.Layer(layer_name="hidden_layer_max_pooling_2d_2", layer_type=tf.keras.layers.MaxPooling2D,
        parameters=[
        ],
        hyperparameters=[
        ]),

    hpo.Layer(layer_name="hidden_layer_conv_2d_2", layer_type=tf.keras.layers.Conv2D,
        parameters=[
          hpo.Parameter(parameter_name="kernel_size", parameter_value=(3, 3)),
          hpo.Parameter(parameter_name="filters", parameter_value=256)
        ],
        hyperparameters=[
        ]),

    hpo.Layer(layer_name="hidden_layer_conv_2d_3", layer_type=tf.keras.layers.Conv2D,
        parameters=[
          hpo.Parameter(parameter_name="kernel_size", parameter_value=(3, 3)),
          hpo.Parameter(parameter_name="filters", parameter_value=256)
        ],
        hyperparameters=[
        ]),

    hpo.Layer(layer_name="hidden_layer_max_pooling_2d_3", layer_type=tf.keras.layers.MaxPooling2D,
        parameters=[
        ],
        hyperparameters=[
        ]),

    hpo.Layer(layer_name="hidden_layer_conv_2d_4", layer_type=tf.keras.layers.Conv2D,
        parameters=[
          hpo.Parameter(parameter_name="kernel_size", parameter_value=(3, 3)),
          hpo.Parameter(parameter_name="filters", parameter_value=512),
            hpo.Parameter(parameter_name="activation", parameter_value="relu")
        ],
        hyperparameters=[
        ]),

    hpo.Layer(layer_name="hidden_layer_conv_2d_5", layer_type=tf.keras.layers.Conv2D,
        parameters=[
          hpo.Parameter(parameter_name="kernel_size", parameter_value=(3, 3)),
          hpo.Parameter(parameter_name="filters", parameter_value=512),
            hpo.Parameter(parameter_name="activation", parameter_value="relu")
        ],
        hyperparameters=[
        ]),

    hpo.Layer(layer_name="hidden_layer_max_pooling_2d_4", layer_type=tf.keras.layers.MaxPooling2D,
        parameters=[
        ],
        hyperparameters=[
        ]),

    hpo.Layer(layer_name="hidden_layer_conv_2d_6", layer_type=tf.keras.layers.Conv2D,
        parameters=[
          hpo.Parameter(parameter_name="kernel_size", parameter_value=(3, 3)),
          hpo.Parameter(parameter_name="filters", parameter_value=512),
            hpo.Parameter(parameter_name="activation", parameter_value="relu")
        ],
        hyperparameters=[
        ]),

    hpo.Layer(layer_name="hidden_layer_conv_2d_7", layer_type=tf.keras.layers.Conv2D,
        parameters=[
          hpo.Parameter(parameter_name="kernel_size", parameter_value=(3, 3)),
          hpo.Parameter(parameter_name="filters", parameter_value=512),
            hpo.Parameter(parameter_name="activation", parameter_value="relu")
        ],
        hyperparameters=[
        ]),

    hpo.Layer(layer_name="hidden_layer_max_pooling_2d_6", layer_type=tf.keras.layers.MaxPooling2D,
        parameters=[
        ],
        hyperparameters=[
        ]),

    hpo.Layer(layer_name="hidden_layer_max_pooling_2d_6", layer_type=tf.keras.layers.MaxPooling2D,
        parameters=[
        ],
        hyperparameters=[
        ]),

    hpo.Layer(layer_name="hidden_layer_flatten", layer_type=tf.keras.layers.Flatten,
        parameters=[
        ],
        hyperparameters=[
        ]),

    hpo.Layer(layer_name="hidden_layer_dense_1", layer_type=tf.keras.layers.Dense,
        parameters=[
            hpo.Parameter(parameter_name="units", parameter_value=4096),
            hpo.Parameter(parameter_name="activation", parameter_value="relu")
        ],
        hyperparameters=[
        ]),

    hpo.Layer(layer_name="hidden_layer_dense_2", layer_type=tf.keras.layers.Dense,
        parameters=[
            hpo.Parameter(parameter_name="units", parameter_value=4096),
            hpo.Parameter(parameter_name="activation", parameter_value="relu")
        ],
        hyperparameters=[
        ]),

    hpo.Layer(layer_name="output_layer_dense", layer_type=tf.keras.layers.Dense,
        parameters=[
            hpo.Parameter(parameter_name="units", parameter_value=150),
            hpo.Parameter(parameter_name="activation", parameter_value="softmax")
        ],
        hyperparameters=[
        ])
]
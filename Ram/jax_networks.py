"""
Jax networks: Contains Haiku code for all the DeepONet models discussed in the paper.
"""

import haiku as hk
import copy
import jax
import jax.numpy as jnp
import os
import sys

from utils import split_conv_string, split_linear_string


def SparseTrunkDeepONetCartesianConvProd(config):
    branch_config = config["branch_config"]
    if branch_config["activation"] == "tanh":
        activation = jax.nn.tanh
    elif branch_config["activation"] == "relu":
        activation = jax.nn.relu
    elif branch_config["activation"] == "elu":
        activation = jax.nn.elu
    elif branch_config["activation"] == "leaky_relu":
        activation = jax.nn.leaky_relu

    trunk_config_list = config["trunk_config_list"]
    p = trunk_config_list[0]["output_dim"]

    def forward(input_, num_groups, num_partitions):
        # param init
        branch_forward = CNN(branch_config)

        func = lambda i, x: hk.switch(i, trunk_applys, x) # evaluates the i'th trunk on the point x and returns the (100,) output
        point_vmap = hk.vmap(func, in_axes=(0, None), split_rng=(not hk.running_init())) # point_vmap([0, 3, 4], x) (3, 100)

        # global set of points
        # [x1]
        # [x2]
        # [x3]
        # [x2207]

        # # group 1
        # [0, x100]
        # [2, x201]
        # [0, x189]
        # [4, x456]
        # [5, x200]

        # # group 2 contains 3 points
        # [[0, 3], x1]
        # [[4, 5], x2]
        # [[0, 2], x3]

        def group_func(i, point, point_weight):
            # i (2,) contains the patch indices a given point belons to
            # point (2,)
            # point_weight (2,) broadcasted to (2,100)
            # point_vmap results in (2, 100) from the two patch trunk networks
            point_eval = point_vmap(i, point) * point_weight # point_weight is already broadcasted
            # point_eval is of shape (2, 100) summed to (100,)
            return point_eval.sum(0)
        group_vmap = hk.vmap(group_func, in_axes=(0, 0, 0), split_rng=(not hk.running_init()))

        bias_param = hk.get_parameter("bias", shape=(1,), init=jnp.zeros)

        trunk_applys = []
        for i in range(num_partitions):
            temp_trunk_forward = MLP(trunk_config_list[i])
            trunk_applys.append(temp_trunk_forward)

        if hk.running_init():
            temp_list = []
            for i in range(num_partitions):
                temp_list.append(trunk_applys[i](input_[1]))

            pu_pred = jnp.stack(temp_list).sum(0)
            branch_pred = branch_forward(input_[0])

            trunk_pred = jnp.hstack([activation(pu_pred)])

            pred = jnp.matmul(branch_pred, trunk_pred.T) + bias_param
            return pred

        # apply
        else:
            branch_pred = branch_forward(input_[0])

            pu_trunk_pred = jnp.zeros((input_[1].shape[0], p)) # (num_points, p) (2207, 100)
            [pu_trunk_pred := pu_trunk_pred.at[input_[4][i]].set(group_vmap(input_[5][i], input_[2][i], input_[3][i])) for i in range(num_groups)]

            trunk_pred = jnp.hstack([activation(pu_trunk_pred)])

            pred = jnp.matmul(branch_pred, trunk_pred.T) + bias_param
            return pred

    return forward


def DeepONetCartesianConvProd(config):
    branch_config = config["branch_config"]
    trunk_config = config["trunk_config"]

    def forward(input_):
        branch_forward = CNN(branch_config)
        trunk_forward = MLP(trunk_config)
        bias_param = hk.get_parameter("bias", shape=(1,), init=jnp.zeros)

        branch_pred = branch_forward(input_[0])
        trunk_pred = trunk_forward(input_[1])
        pred = jnp.matmul(branch_pred, trunk_pred.T) + bias_param
        return pred

    return forward


def CNN(config):
    if config["activation"] == "tanh":
        activation = jax.nn.tanh
    elif config["activation"] == "relu":
        activation = jax.nn.relu
    elif config["activation"] == "elu":
        activation = jax.nn.elu
    elif config["activation"] == "leaky_relu":
        activation = jax.nn.leaky_relu

    def forward(x):
        layers_ = []
        for i in range(len(config["layers"])):
            if "conv" in config["layers"][i]:
                _, num_channels, kernel_size, stride, padding = split_conv_string(config["layers"][i])
                layers_.append(hk.Conv2D(output_channels=num_channels, kernel_shape=kernel_size, stride=stride, padding=padding, name=config["name"]+f"_conv_{i}"))
            elif config["layers"][i] == "activation":
                layers_.append(activation)
            elif config["layers"][i] == "flatten":
                layers_.append(hk.Flatten())
            elif "linear" in config["layers"][i]:
                _, num_neurons = split_linear_string(config["layers"][i])
                layers_.append(hk.Linear(num_neurons, name=config["name"]+f"_linear_{i}"))
            else:
                raise f"Layer {config['layers'][i]} not configured"

        cnn = hk.Sequential(layers_)
        return cnn(x)

    return forward


def MLP(config):
    if config["activation"] == "tanh":
        activation = jax.nn.tanh
    elif config["activation"] == "relu":
        activation = jax.nn.relu
    elif config["activation"] == "elu":
        activation = jax.nn.elu
    elif config["activation"] == "leaky_relu":
        activation = jax.nn.leaky_relu

    if config.get("layer_sizes", None) is None:
        hidden_layers = [config["nodes"] for _ in range(config["num_hidden_layers"])]
        if config["nodes"] == 0 or config["num_hidden_layers"] == 0:
            layer_sizes = [config["output_dim"]]
        else:
            layer_sizes = hidden_layers + [config["output_dim"]]
    else:
        hidden_layers = config["layer_sizes"]
        layer_sizes = hidden_layers + [config["output_dim"]]

    def forward(x):
        mlp_module = hk.nets.MLP(output_sizes=layer_sizes, with_bias=config.get("use_bias", True), activation=activation, activate_final=config.get("last_layer_activate", False), name=config["name"])
        return mlp_module(x)

    return forward


def Linear(output_dim, use_bias=True):
    def forward(x):
        linear_module = hk.Linear(output_dim, with_bias=use_bias)
        return linear_module(x)

    return forward


def get_model(model_name, config):
    _MODELS = dict(
        mlp=MLP,
        linear=Linear,
        cnn=CNN,
        deeponet_cartesian_conv_prod=DeepONetCartesianConvProd,
        sparse_trunk_deeponet_cartesian_conv_prod=SparseTrunkDeepONetCartesianConvProd
    )

    if model_name not in _MODELS.keys():
        raise NameError('Available keys:', _MODELS.keys())

    # initialize with config
    net_fn = _MODELS[model_name](config)

    net = hk.transform(net_fn)
    return net.init, net.apply


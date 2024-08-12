import torch
import copy
from scipy import io
from scipy.io import savemat, loadmat
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle
import jax
import jax.numpy as jnp
import numpy as onp
import os
from functools import partial
import sys
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from train_utils import train_func
from utils import fstr, save_results, PU
from jax_networks import get_model


def get_data(filename):
    """
    Please refer to the paper for the datasets used.
    """
    try:
        gamma = io.loadmat(filename+"gamma.mat")["gamma_vec"]
    except:
        raise BaseException("\n\nPlease place the gamma.mat file in the dataset/diffrec folder.")
    gamma = jnp.asarray(gamma).astype(dtype)
    X = io.loadmat(filename+"X.mat")["X"]
    c_end = jnp.asarray(io.loadmat(filename+"c_end.mat")["c1"]).T

    trunk_input = jnp.asarray(X).astype(dtype)
    gamma_broadcast = gamma[:, jnp.newaxis, jnp.newaxis] * jnp.ones((1, m, m))
    gamma_broadcast = gamma_broadcast.reshape(-1, m, m, 1)
    x_branch_train = gamma_broadcast[:N, :]
    x_branch_test = gamma_broadcast[N:N+N_test, :]

    x_branch_train = x_branch_train.astype(dtype)
    x_branch_test = x_branch_test.astype(dtype)
    trunk_input = trunk_input.astype(dtype)
    y_train = c_end[:N, :].astype(dtype)
    y_test = c_end[N:N+N_test, :].astype(dtype)

    train_input = (x_branch_train, trunk_input)
    test_input = (x_branch_test, trunk_input)

    return train_input, y_train, test_input, y_test


def vanilla_deeponet_run():
    project = "vanilla_deeponet"
    model_name = "deeponet_cartesian_conv_prod"
    save_file_name = fstr(save_folder.payload + "{project}_p={p}_seed={seed}.json")
    param_save_file_name = fstr(save_folder.payload + "{project}_p={p}_seed={seed}.pickle")
    log_str = fstr("vanilla-deeponet on {problem} with {opt_choice} p={p} seed={seed}")

    trunk_input = 2
    branch_input = m

    for seed in seeds:
        for p in ps:
            trunk_output = p
            branch_output = p

            local_branch_layer_sizes = copy.deepcopy(branch_layer_sizes)
            local_branch_layer_sizes.append(f"linear_{branch_output}")

            trunk_config = dict(
                activation=activation_choice,
                last_layer_activate=True,
                layer_sizes=trunk_layer_sizes,
                output_dim=trunk_output,
                name="trunk"
            )

            branch_config = dict(
                activation=activation_choice,
                layers=local_branch_layer_sizes,
                output_dim=branch_output,
                name="branch"
            )

            bias_config = dict(name="bias")

            model_config = dict(
                branch_config=branch_config,
                trunk_config=trunk_config,
                bias_config=bias_config
            )

            if not os.path.isdir(str(save_folder)):
                os.makedirs(str(save_folder))

            print("\n\n")
            print("+"*100)
            print(log_str)
            print(str(save_folder))
            print("+"*100)
            print("\n\n")

            train_input, Y, test_input, Y_test = get_data(str(dataset_folder))

            train_input = (train_input[0].reshape(-1, m, m, 1), train_input[1])
            test_input = (test_input[0].reshape(-1, m, m, 1), test_input[1])

            dummy_input = (jnp.expand_dims(train_input[0][0, :, :, :], 0), jnp.expand_dims(train_input[1][0, :], 0))

            hyperparameter_dict = dict(
                print_bool=print_bool,
                print_interval=print_interval,
                epochs=epochs,
                model_config=model_config,
                opt_choice=opt_choice,
                schedule_choice=schedule_choice,
                lr_dict=lr_dict,
                problem=problem,
                N=N,
                N_test=N_test,
                p=p,
                m=m,
                dtype=dtype,
                seed=seed
            )

            train_config = dict(
                dummy_input=dummy_input,
                train_input=train_input,
                test_input=test_input,
                Y=Y,
                Y_test=Y_test,
                model_name=model_name,
            ) | hyperparameter_dict

            logged_results, trained_params = train_func(train_config)
            logged_results = logged_results | hyperparameter_dict

            if save_results_bool:
                torch.save(trained_params, str(param_save_file_name))
                save_results(logged_results, str(save_file_name))


def sparse_trunk_deeponet_run():
    project = "sparse_trunk_deeponet"
    model_name = "sparse_trunk_deeponet_cartesian_conv_prod"

    save_file_name = fstr(save_folder.payload + "{project}_M={M}_p={p}_seed={seed}.json")
    param_save_file_name = fstr(save_folder.payload + "{project}_M={M}_p={p}_seed={seed}.pickle")
    log_str = fstr("sparse-trunk-deeponet on {problem} with {opt_choice} M={M} p={p} seed={seed}")

    trunk_input = 2
    branch_input = m

    for seed in seeds:
        for p in ps:
            trunk_output = p
            branch_output = p

            local_branch_layer_sizes = copy.deepcopy(branch_layer_sizes)
            local_branch_layer_sizes.append(f"linear_{branch_output}")

            branch_config = dict(
                activation=activation_choice,
                layers=local_branch_layer_sizes,
                output_dim=branch_output,
                name="branch"
            )

            bias_config = dict(name="bias")

            if not os.path.isdir(str(save_folder)):
                os.makedirs(str(save_folder))

            print("\n\n")
            print("+"*100)
            print(log_str)
            print(str(save_folder))
            print("+"*100)
            print("\n\n")

            train_input, Y, test_input, Y_test = get_data(str(dataset_folder))

            train_input = (train_input[0].reshape(-1, m, m, 1), train_input[1])
            test_input = (test_input[0].reshape(-1, m, m, 1), test_input[1])

            # partitioning ============================================================
            # TRAINING
            train_trunk_input = train_input[1]
            test_trunk_input = test_input[1]

            pu_config = dict(
                num_partitions=M,
                dim=train_trunk_input.shape[-1],
            )

            pu_obj = PU(pu_config)

            pu_obj.partition_domain(train_trunk_input, centers=centers_list, radius=radius_list)
            print("Radius = ", pu_obj.radius)
            print(f"M={M}")
            print("Checking train partitioning")
            K, M_changed, _ = pu_obj.check_partioning(train_trunk_input, change_M=True, min_partitions_per_point=min_partitions_per_point)
            pu_trunk_width = int(1/onp.sqrt(K) * trunk_width)
            print(f"K={K}, PU trunk_width={pu_trunk_width}")
            print(f"new M={M_changed}")
            print("Forming points per groups")
            num_groups, participation_idx, points_per_group, indices_per_group, _, radius_arrs = pu_obj.form_points_per_group(train_trunk_input)
            print("Group sizes: ", [len(g) for g in points_per_group])
            print("Forming weights per groups")
            weights_per_group = pu_obj.form_weights_per_group(points_per_group=points_per_group, participation_idx=participation_idx, radius_arrs=radius_arrs)
            weights_per_group = [jnp.broadcast_to(w.reshape(*w.shape, 1), (w.shape[0], w.shape[1], p)) for w in weights_per_group]

            # TEST
            print("\nChecking test partitioning")
            _, M_test, _ = pu_obj.check_partioning(test_trunk_input, False, min_partitions_per_point=min_partitions_per_point)
            print(f"M_test={M}")
            print("Forming points per groups")
            test_num_groups, test_participation_idx, test_points_per_group, test_indices_per_group, _, test_radius_arrs = pu_obj.form_points_per_group(test_trunk_input)
            print("Group sizes: ", [len(g) for g in test_points_per_group])
            print("Forming weights per groups")
            test_weights_per_group = pu_obj.form_weights_per_group(points_per_group=test_points_per_group, participation_idx=test_participation_idx, radius_arrs=test_radius_arrs)
            test_weights_per_group = [jnp.broadcast_to(w.reshape(*w.shape, 1), (w.shape[0], w.shape[1], p)) for w in test_weights_per_group]

            # reassembling training and test and dummy tuples
            train_input = (train_input[0], train_input[1], points_per_group, weights_per_group, indices_per_group, participation_idx, num_groups)
            test_input = (test_input[0], test_input[1], test_points_per_group, test_weights_per_group, test_indices_per_group, test_participation_idx, test_num_groups)
            dummy_input = (jnp.expand_dims(train_input[0][0, :], 0), jnp.expand_dims(train_input[1][0, :], 0))

            pu_trunk_config = dict(
                last_layer_activate=False,
                input_dim=trunk_input,
                output_dim=trunk_output,
                name="pu_trunk"
            )

            trunk_config_list = []
            for trunk_idx in range(M_changed):
                temp_config = copy.deepcopy(pu_trunk_config)
                temp_config["activation"] = activation_choice
                temp_config["name"] = f"pu_{trunk_idx}_trunk"
                if trunk_idx < trunk_special_index:
                    temp_config["layer_sizes"] = trunk_layer_sizes
                else:
                    temp_config["layer_sizes"] = [trunk_width for _ in range(trunk_depth)]
                trunk_config_list.append(temp_config)

            model_config = dict(
                branch_config=branch_config,
                trunk_config_list=trunk_config_list,
                bias_config=bias_config,
                num_partitions=M_changed
            )

            hyperparameter_dict = dict(
                print_bool=print_bool,
                print_interval=print_interval,
                epochs=epochs,
                model_config=model_config,
                opt_choice=opt_choice,
                schedule_choice=schedule_choice,
                lr_dict=lr_dict,
                problem=problem,
                N=N,
                N_test=N_test,
                p=p,
                M=M_changed,
                K=K,
                train_num_partitions=M_changed,
                test_num_partitions=M_test,
                train_num_groups=num_groups,
                test_num_groups=test_num_groups,
                m=m,
                dtype=dtype,
                seed=seed
            )

            train_config = dict(
                dummy_input=dummy_input,
                train_input=train_input,
                test_input=test_input,
                Y=Y,
                Y_test=Y_test,
                model_name=model_name,
            ) | hyperparameter_dict

            logged_results, trained_params = train_func(train_config)
            logged_results = logged_results | hyperparameter_dict

            if save_results_bool:
                torch.save(trained_params, str(param_save_file_name))
                save_results(logged_results, str(save_file_name))


if __name__ == "__main__":
    dataset_folder = fstr("./dataset/")

    save_results_bool = False
    print_bool = True

    problem = "diffrec"
    opt_choice = "adam"
    schedule_choice = "inverse_time_decay"
    # activations
    activation_choice = "relu"  # activation everywhere

    print_interval = 100
    dtype = "float32"

    N = 1000
    N_test = 200
    m = 20

    lr_dict = dict(
        peak_lr=3e-4,
        weight_decay=1e-4,
        decay_steps=1,
        decay_rate=1e-4,
    )

    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    else:
        jax.config.update("jax_enable_x64", False)

    epochs = 150000

    centers_list = [
        [1., 1.5],
        [1., 0.5],

        [.3, 1.5],
        [.3, 0.5],

        [1.7, 1.5],
        [1.7, 0.5],
    ]

    M = len(centers_list)

    radius_list = []
    cnt = 1
    for c in centers_list:
        if cnt <= 2:
            radius_list.append(0.6)
        else:
            radius_list.append(0.65)
        cnt += 1
    trunk_special_index = M

    # sub project folder
    all_run_folder = "deeponet_results"

    # architectures ===========================================================================
    trunk_depth = 3
    trunk_width = 128
    trunk_layer_sizes = [trunk_width for _ in range(trunk_depth)]

    branch_layer_sizes = [
        "conv_64_5_2_valid",
        "activation",
        "conv_128_5_2_valid",
        "activation",
        "flatten",
        "linear_128",
        "activation"
    ]

    seeds = [0, 1, 2, 3, 4]
    ps = [100]
    # ============================================================================================================
    # vanilla deeponet

    folder_string = f"vanilla_deeponet_results"
    save_folder = fstr("./{all_run_folder}/{folder_string}/{problem}/")
    # vanilla_deeponet_run()

    # ============================================================================================================
    # sparse trunk deeponet

    min_partitions_per_point = 1
    folder_string = f"sparse_trunk_deeponet_results"
    save_folder = fstr("./{all_run_folder}/{folder_string}/{problem}/")
    sparse_trunk_deeponet_run()


"""
Training utils: A generic Jax training function for DeepONets.
"""

from collections import defaultdict
import jax
import jax.numpy as jnp
import time
import json
import optax
from jax.example_libraries import optimizers
from functools import partial
import os
import sys

sys.path.append((os.path.dirname(os.path.dirname(__file__))))
from jax_networks import get_model


def train_func(train_info):
    # to store results
    logged_results = defaultdict(list)

    @partial(jax.jit, static_argnums=(3, 4))
    def pu_loss(all_params, input_, y, num_groups, num_partitions):
        pred = model_forward(all_params, model_key, input_, num_groups, num_partitions)
        return jnp.power(pred - y, 2).mean()

    @partial(jax.jit, static_argnums=(4, 5))
    def pu_step(all_params, opt_state, input_, y, num_groups, num_partitions):
        # grads is for all grads
        loss_val, grads = jax.value_and_grad(pu_loss)(all_params, input_, y, num_groups, num_partitions)
        updates, opt_state = opt.update(grads, opt_state, all_params)
        all_params = optax.apply_updates(all_params, updates)
        return all_params, opt_state, loss_val

    @jax.jit
    def loss(all_params, input_, y):
        pred = model_forward(all_params, model_key, input_)
        return jnp.power(pred - y, 2).mean()

    @jax.jit
    def step(all_params, opt_state, input_, y):
        # grads is for all grads
        loss_val, grads = jax.value_and_grad(loss)(all_params, input_, y)
        updates, opt_state = opt.update(grads, opt_state, all_params)
        all_params = optax.apply_updates(all_params, updates)
        return all_params, opt_state, loss_val

    # hyperparameters and dataset
    print_interval = train_info["print_interval"]
    print_bool = train_info["print_bool"]
    epochs = train_info["epochs"]
    train_input = train_info["train_input"]
    test_input = train_info["test_input"]
    Y = train_info["Y"]
    Y_test = train_info["Y_test"]
    dummy_input = train_info["dummy_input"]
    sparse_bool = "sparse" in train_info["model_name"] or "ensemble_sparse" in train_info["model_name"]

    model_key = jax.random.PRNGKey(train_info["seed"])

    # model choice
    if not sparse_bool:
        model_init, model_forward = get_model(train_info["model_name"], train_info["model_config"])
        model_forward = jax.jit(model_forward)
        # initializing model parameters
        all_params = model_init(model_key, dummy_input)
    else:
        model_key = jax.random.PRNGKey(train_info["seed"])
        model_init, model_forward = get_model(train_info["model_name"], train_info["model_config"])
        model_forward = jax.jit(model_forward, static_argnums=(3, 4))

        train_num_groups = train_info["train_num_groups"]
        test_num_groups = train_info["test_num_groups"]
        train_num_partitions = train_info["train_num_partitions"]
        test_num_partitions = train_info["test_num_partitions"]
        # initializing model parameters
        all_params = model_init(model_key, dummy_input, train_num_groups, train_num_partitions)

    if train_info["schedule_choice"] == "warmup_cosine_decay":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=train_info["lr_dict"]["init_lr"],
            peak_value=train_info["lr_dict"]["peak_lr"],
            warmup_steps=train_info["lr_dict"]["warmup_steps"],
            decay_steps=train_info["lr_dict"]["decay_steps"]
        )
    elif train_info["schedule_choice"] == "inverse_time_decay":
        schedule = optimizers.inverse_time_decay(
            step_size=train_info["lr_dict"]["peak_lr"],
            decay_steps=train_info["lr_dict"]["decay_steps"],
            decay_rate=train_info["lr_dict"]["decay_rate"],
            staircase=train_info["lr_dict"].get("staircase", False)
        )
    elif train_info["schedule_choice"] == "no_schedule":
        schedule = train_info["peak_lr"]
    else:
        raise "Unknown learning rate schedule"

    if train_info["opt_choice"] == "adam":
        opt = optax.adam(learning_rate=schedule)
    elif train_info["opt_choice"] == "adamw":
        opt = optax.adamw(learning_rate=schedule, weight_decay=train_info["lr_dict"]["weight_decay"])

    # initializing optimizer state with parameters
    opt_state = opt.init(all_params)

    # warmup step
    if not sparse_bool:
        _, _, _ = step(all_params, opt_state, train_input, Y)
    else:
        _, _, _ = pu_step(all_params, opt_state, train_input, Y, train_num_groups, train_num_partitions)

    for i in range(epochs):
        if not sparse_bool:
            start = time.perf_counter()
            all_params, opt_state, loss_val = step(all_params, opt_state, train_input, Y)
            epoch_time = time.perf_counter() - start
        else:
            start = time.perf_counter()
            all_params, opt_state, loss_val = pu_step(all_params, opt_state, train_input, Y, train_num_groups, train_num_partitions)
            epoch_time = time.perf_counter() - start

        logged_results["training_loss"].append(loss_val.item())
        logged_results["training_epoch_time"].append(epoch_time)

        if i % print_interval == 0 and print_bool:
            print("="*15)
            print(f"Epoch {i}:")
            print(f"Loss: {loss_val}")
            print(f"Time: {epoch_time}")

    # train inference warmup
    if not sparse_bool:
        _ = model_forward(all_params, None, train_input)
        start = time.perf_counter()
        train_pred = model_forward(all_params, None, train_input)
        train_inference_time = time.perf_counter() - start
    else:
        _ = model_forward(all_params, None, train_input, train_num_groups, train_num_partitions)
        start = time.perf_counter()
        train_pred = model_forward(all_params, None, train_input, train_num_groups, train_num_partitions)
        train_inference_time = time.perf_counter() - start

    # test inference warmup
    if not sparse_bool:
        _ = model_forward(all_params, None, test_input)
        start = time.perf_counter()
        test_pred = model_forward(all_params, None, test_input)
        test_inference_time = time.perf_counter() - start
    else:
        _ = model_forward(all_params, None, test_input, test_num_groups, test_num_partitions)
        start = time.perf_counter()
        test_pred = model_forward(all_params, None, test_input, test_num_groups, test_num_partitions)
        test_inference_time = time.perf_counter() - start

    assert train_pred.shape == Y.shape, f"train pred = {train_pred.shape}, Y = {Y.shape}"
    assert test_pred.shape == Y_test.shape, f"test pred = {test_pred.shape}, Y_test = {Y_test.shape}"

    if len(test_pred.shape) == 2:
        test_l2_error = jnp.mean(jnp.linalg.norm(test_pred - Y_test, axis=1) / jnp.linalg.norm(Y_test, axis=1))
        test_linf_error = jnp.mean(jnp.linalg.norm(test_pred - Y_test, axis=1, ord=jnp.inf) / jnp.linalg.norm(Y_test, axis=1, ord=jnp.inf))
        train_l2_error = jnp.mean(jnp.linalg.norm(train_pred - Y, axis=1) / jnp.linalg.norm(Y, axis=1))
        train_linf_error = jnp.mean(jnp.linalg.norm(train_pred - Y, axis=1, ord=jnp.inf) / jnp.linalg.norm(Y, axis=1, ord=jnp.inf))
    elif len(test_pred.shape == 3):
        test_pred = jnp.linalg.norm(test_pred, axis=2)
        train_pred = jnp.linalg.norm(train_pred, axis=2)
        Y = jnp.linalg.norm(Y, axis=2)
        Y_test = jnp.linalg.norm(Y_test, axis=2)
        test_l2_error = jnp.mean(jnp.linalg.norm(test_pred - Y_test, axis=1) / jnp.linalg.norm(Y_test, axis=1))
        test_linf_error = jnp.mean(jnp.linalg.norm(test_pred - Y_test, axis=1, ord=jnp.inf) / jnp.linalg.norm(Y_test, axis=1, ord=jnp.inf))
        train_l2_error = jnp.mean(jnp.linalg.norm(train_pred - Y, axis=1) / jnp.linalg.norm(Y, axis=1))
        train_linf_error = jnp.mean(jnp.linalg.norm(train_pred - Y, axis=1, ord=jnp.inf) / jnp.linalg.norm(Y, axis=1, ord=jnp.inf))

    test_mse_error = jnp.power(test_pred.flatten() - Y_test.flatten(), 2).mean()
    train_mse_error = jnp.power(train_pred.flatten() - Y.flatten(), 2).mean()

    logged_results["train_inference_time"] = train_inference_time
    logged_results["train_l2_error"] = train_l2_error.item()
    logged_results["train_linf_error"] = train_linf_error.item()
    logged_results["train_mse_error"] = train_mse_error.item()
    logged_results["test_inference_time"] = test_inference_time
    logged_results["test_l2_error"] = test_l2_error.item()
    logged_results["test_linf_error"] = test_linf_error.item()
    logged_results["test_mse_error"] = test_mse_error.item()

    if print_bool:
        print(f"Train relative L2 error: {train_l2_error}")
        print(f"Test relative L2 error: {test_l2_error}")
        print(f"Train relative Linf error: {train_linf_error}")
        print(f"Test relative Linf error: {test_linf_error}")
        print(f"Train MSE: {train_mse_error}")
        print(f"Test MSE: {test_mse_error}")
        print(f"Train inference time: {train_inference_time}")
        print(f"Test inference time: {test_inference_time}")

    return logged_results, all_params

import jax
import optax
import haiku as hk
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

# Load and preprocess the MNIST datasetppPp
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to range [0, 1] and convert to float32
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Convert numpy arrays to JAX arrays
x_train = jnp.array(x_train)
x_val = jnp.array(x_val)
y_train = jnp.array(y_train)
y_val = jnp.array(y_val)

print(x_train.shape)
print(y_train.shape)

print(type(x_train))
print(type(y_train))



class MNIST_model(hk.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def __call__(self, x):
        x = hk.Conv2D(output_channels=32, kernel_shape=3, stride=1, padding='SAME')(x)
        x = jax.nn.relu(x)
        x = hk.MaxPool(window_shape=2, strides=1, padding='SAME')(x)

        x = hk.Conv2D(output_channels=64, kernel_shape=3, stride=1, padding='SAME')(x) # increase the number of channels bcz dense layers learn more precise features
        x = jax.nn.relu(x)
        x = hk.MaxPool(window_shape=2, strides=1, padding='SAME')(x)

        x = hk.Flatten()(x) # converting the data into a single column

        x = hk.Linear(64)(x)
        x = jax.nn.relu(x)

        x = hk.Linear(64)(x)
        x = jax.nn.relu(x)

        x = hk.Linear(self.num_classes)(x)
        x = jax.nn.softmax(x)  # Apply softmax activation
        return x
    
def forward_fn(x):
    model = MNIST_model(num_classes = 10) # since there are 10 different numbers
    return model(x)

forward = hk.transform(forward_fn)

rng = jax.random.PRNGKey(42)
x_sample = x_train[:1]
params = forward.init(rng, x_sample)

def loss_fn(params, x, y):
    epsilon = 1e-7
    predictions = forward.apply(params, None, x)
    predictions = jnp.clip(predictions, epsilon, 1.0 - epsilon)
    batch_sz = predictions.shape[0]
    log_probs = jnp.log(predictions[jnp.arange(batch_sz), y.astype(int)])
    loss = -jnp.mean(log_probs)
    return loss


# Initialize optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def update(params, opt_state, x, y):
    grads = jax.grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    return new_params, opt_state

def print_weights(params):
    for i, param in enumerate(jax.tree_leaves(params)):
        print(f"Weights {i}: {param}")

num_epochs = 5
batch_size = 64
num_batches = x_train.shape[0]//batch_size
print(num_batches)
for epoch in range(num_epochs):
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        x_batch = x_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        # print(f"start: {start_idx}")
        # print(f"end: {end_idx}")
        params, opt_state = update(params, opt_state, x_batch, y_batch)

        if batch_idx % 200 == 0:
            prediction1 = forward.apply(params,None,x_batch)
            accuracy1 = jnp.mean(jnp.argmax(prediction1, axis=-1) == y_batch)
            print(f"Training Accuracy: {accuracy1}")

            train_loss = loss_fn(params, x_batch, y_batch)
            val_loss = loss_fn(params, x_val, y_val)

            prediction2 = forward.apply(params,None,x_val)
            accuracy2 = jnp.mean(jnp.argmax(prediction2, axis=-1) == y_val)
            print(f"Validation Accuracy: {accuracy2}")

            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{num_batches}, Train Loss: {train_loss}, Val Loss: {val_loss}")

            # Print the weights
            # print_weights(params)
            print("-----------------")

test_loss = loss_fn(params, x_test, y_test)
print(f"Test Loss: {test_loss}")

num_samples = 5  # Number of test cases to show predictions for
for i in range(num_samples):
    x_sample = x_test[i:i+1]
    y_true = y_test[i]
    logits = forward.apply(params, None, x_sample)
    prediction = jnp.argmax(logits, axis=-1)[0]
    print(f"Sample {i+1}: Prediction = {prediction}, True Label = {y_true}")

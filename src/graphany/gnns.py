from functools import partial

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class MLP(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        nlayers: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.nlayers = nlayers

        self.linear_in = nnx.Linear(input_dim, hidden_dim, rngs=rngs)

        @nnx.split_rngs(split=nlayers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layers(rngs: nnx.Rngs):
            return nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)

        self.layers = create_layers(rngs)
        self.linear_out = nnx.Linear(hidden_dim, output_dim)

    def __call__(self, x: jax.Array):
        def forward(x: jax.Array, layer: nnx.Module):
            x = nnx.leaky_relu(x)
            return layer(x)

        y = self.linear_in(x)
        y, _ = jax.lax.scan(
            forward, init=y, xs=self.layers, length=self.nlayers
        )
        y = nnx.leaky_relu(y)
        y = self.linear_out(y)
        return y


class LinearGNN(nnx.Module):
    def __init__(self, nchannels: int = 2):
        self.nchannels = nchannels

    def __call__(
        self, adj: jax.Array, x: jax.Array, y: jax.Array, train_mask: jax.Array
    ):
        _, num_nodes, _ = adj.shape

        # We convolve the features
        def matrix_power(carry: jax.Array, _, adj: jax.Array):
            carry = adj @ carry
            return carry, carry

        # Compute the positive and negative convolutions
        degrees = adj.sum(axis=(2,), keepdims=True)
        dgn_adj = adj / degrees

        _, pos = jax.lax.scan(
            partial(matrix_power, adj=dgn_adj),
            init=x,
            length=self.nchannels,
        )
        _, neg = jax.lax.scan(
            partial(matrix_power, adj=jnp.eye(num_nodes) - dgn_adj),
            init=x,
            length=self.nchannels,
        )

        # Compute the least squares predictions
        convolved_x = jnp.concatenate(
            [x[None], pos, neg], axis=0
        )  # (2 * nchannels + 1, batch_size, num_nodes, d)

        # We evaluate the least squares solution to compute class-sized node-wise features
        x_train = convolved_x * train_mask[None, ..., None]
        y_train = y * train_mask[..., None]

        def fit_regression(xc: jax.Array, yc: jax.Array):
            return jnp.linalg.pinv(xc) @ yc

        lstsq_val = jax.vmap(
            jax.vmap(fit_regression, in_axes=(0, 0), out_axes=0),
            in_axes=(0, None),
            out_axes=0,
        )(x_train, y_train)  # (2 * nchannels + 1, d, num_classes)
        features = jnp.matmul(
            convolved_x, lstsq_val
        )  # (2 * nchannels + 1, batch_size, num_nodes, num_classes)
        return features


if __name__ == "__main__":
    m = LinearGNN()
    d = 32
    c = 6
    key = jax.random.key(42)

    adj = jnp.array(
        [
            [[0, 1, 0], [0, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        ]
    )
    x = jax.random.normal(key, (2, 3, d))
    y = jax.nn.one_hot(
        jax.random.categorical(key, jnp.ones(c), shape=(2, 3)),
        num_classes=c,
    )

    print(m(adj, x, y, train_mask=jnp.ones((2, 3))).shape)

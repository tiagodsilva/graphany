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

        @nnx.split_rngs(splits=nlayers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layers(rngs: nnx.Rngs):
            return nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)

        self.layers = create_layers(rngs)
        self.linear_out = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x: jax.Array):
        def forward(x: jax.Array, layer: nnx.Module):
            x = nnx.leaky_relu(x)
            return layer(x), None

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
        dgn_adj = jnp.where(degrees == 0, 0, adj / degrees)

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


class GraphAny(nnx.Module):
    def __init__(
        self,
        nchannels: int = 2,
        hidden_dim: int = 32,
        entropy: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.entropy = entropy
        self.linear = LinearGNN(nchannels)

        self.all_channels = 2 * nchannels + 1
        self.attn = MLP(
            self.all_channels * (self.all_channels - 1),
            hidden_dim,
            self.all_channels,
            rngs=rngs,
        )

    def normalize(self, x: jax.Array):
        dist = jnp.power(x[..., None] - x[..., None, :], 2).sum(
            axis=-3
        )  # (batch_size, num_nodes, self.all_channels, self.all_channels)

        # Reshape the distances
        eye_mask = jnp.eye(self.all_channels, dtype=bool)
        p_ij = dist[
            ..., ~eye_mask
        ]  # (batch_size, num_nodes, self.all_channels * (self.all_channels  - 1))

        # Compute the normalized features
        batch_size, num_nodes, _ = p_ij.shape
        p_ij = p_ij.reshape(
            batch_size, num_nodes, self.all_channels, self.all_channels - 1
        )

        def cal_entropy(carry: jax.Array, _):
            sigma, sigma_min, sigma_max = carry
            psigma = -p_ij / (2 * sigma[..., None] ** 2)
            e = -jnp.sum(
                nnx.softmax(psigma, axis=-1)
                * nnx.log_softmax(psigma, axis=-1),
                axis=-1,
            )

            sigma_next = jnp.where(
                e > self.entropy,
                sigma - (sigma - sigma_min) / 2,
                sigma + (sigma_max - sigma) / 2,
            )
            sigma_min_next = jnp.where(e > self.entropy, sigma_min, sigma)
            sigma_max_next = jnp.where(e > self.entropy, sigma, sigma_max)

            return (sigma_next, sigma_min_next, sigma_max_next), e

        sigma = jnp.ones((batch_size, num_nodes, self.all_channels))

        (sigma, _, _), _ = jax.lax.scan(
            cal_entropy,
            init=(sigma, sigma * 1e-3, sigma * 1e3),
            length=int(1e2),
        )

        features = nnx.softmax(-p_ij / (2 * sigma[..., None] ** 2), axis=-1)
        return features.reshape(batch_size, num_nodes, -1)

    def __call__(
        self, adj: jax.Array, x: jax.Array, y: jax.Array, train_mask: jax.Array
    ):
        # Compute the linear features
        linear_features = self.linear(
            adj, x, y, train_mask
        )  # (self.all_channels, batch_size, num_nodes, num_classes)
        linear_features = jnp.transpose(
            linear_features, (1, 2, 3, 0)
        )  # (batch_size, num_nodes, num_classes, self.all_channels)

        attn_val = self.normalize(linear_features)
        attn_val = nnx.softmax(
            self.attn(attn_val), axis=-1
        )  # (batch_size, num_nodes, self.all_channels)

        return jnp.sum(attn_val[..., None, :] * linear_features, axis=-1)


if __name__ == "__main__":
    m = LinearGNN()
    g = GraphAny(rngs=nnx.Rngs(42))
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

    train_mask = jnp.ones((2, 3))
    print(m(adj, x, y, train_mask).shape)
    print(g(adj, x, y, train_mask).shape)

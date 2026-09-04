from functools import partial

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import tqdm
from gcsbm.csbm import NULL_LABEL, CSBMParamPrior, simulate

from graphany.gnns import GraphAny


def create_opt(model: nnx.Module, lr: float = 1e-3):
    optimizer = optax.adam(lr)
    return nnx.Optimizer(model, optimizer, wrt=nnx.Param)


def train_on_csbm(
    model: GraphAny,
    opt: nnx.Optimizer,
    iterations: int = int(1e3),
    seed: int = 42,
    batch_size: int = 8,
    num_nodes: int = 32,
    num_labels: int = 2,
    num_features: int = 6,
):
    key = jax.random.key(seed)

    theta_prior = CSBMParamPrior.non_informative(
        num_labels, num_features=num_features
    )

    @nnx.jit
    def loss_fn(model, adj, y, x, train_mask):
        preds = model(
            adj, x, y, train_mask
        )  # (batch_size, num_nodes, num_classes)

        safe_preds = jnp.where(y == 1, preds, 1.0)
        loss = -jnp.log(safe_preds).sum(axis=-1).mean()
        return loss

    simulate_vmap = jax.vmap(
        partial(
            simulate, theta_prior=theta_prior, num_nodes=num_nodes, sigma=2.0
        ),
        in_axes=(0,),
        out_axes=0,
    )

    def to_onehot(labels: jax.Array):
        train_mask = y != NULL_LABEL
        return jax.nn.one_hot(y, num_classes=num_labels), train_mask

    # Training
    for _ in (pbar := tqdm.trange(iterations)):
        key, *subkeys = jax.random.split(key, batch_size + 1)
        subkeys = jnp.stack(subkeys)

        # Simulate the data
        (adj, y, x), _ = simulate_vmap(subkeys)
        y, train_mask = to_onehot(y)

        # Compute the loss function and update the model
        loss, grads = nnx.value_and_grad(loss_fn, argnums=0)(
            model, adj, y, x, train_mask
        )
        opt.update(model, grads)

        pbar.set_postfix(loss=f"{loss:.2e}")

    # Inference
    keys = jax.random.split(key, batch_size)
    key = jnp.stack(keys)

    (adj, y, x), (true_labels, _, _) = simulate_vmap(keys)
    y, train_mask = to_onehot(y)
    preds = model(adj, x, y, train_mask)
    preds = jnp.argmax(preds, axis=-1)
    acc = (preds == true_labels)[~train_mask].mean()

    print(acc)

    return model


if __name__ == "__main__":
    rngs = nnx.Rngs(42)
    model = GraphAny(rngs=rngs)
    opt = create_opt(model)

    train_on_csbm(model, opt)

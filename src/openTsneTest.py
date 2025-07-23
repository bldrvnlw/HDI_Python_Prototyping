from utils.data_sources import get_generated, get_MNIST, get_mouse_Zheng
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils.prob_utils import (
    compute_annoy_probabilities,
    euclidian_sqrdistance_matrix,
    compute_perplexity_probs_numba,
    symmetrize_probs,
    get_random_uniform_circular_embedding,
)
from openTSNE import TSNE

num_points = 50000
X, y, colors, unique_colors = get_mouse_Zheng(num_points=num_points)
points = get_random_uniform_circular_embedding(num_points, 0.1)


def callback(iter, kldiv, embedding):
    print(f"KL divergence {kldiv}")


tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
    neighbors="annoy",
    negative_gradient_method="interpolation",
    callbacks_every_iters=1,
    initialization="random",
    callbacks=[callback],
)

embedding_train = tsne.fit(X)

scatter = plt.scatter(embedding_train[:, 0], embedding_train[:, 1], c=colors, alpha=0.7)
custom = [
    Line2D(
        [],
        [],
        marker=".",
        markersize=len(unique_colors),
        color=unique_colors[i],
        linestyle="None",
    )
    for i in range(0, len(unique_colors))
]
plt.legend(
    custom, [x for x in range(0, len(unique_colors))], loc="lower left", title="Cluster"
)
plt.show()

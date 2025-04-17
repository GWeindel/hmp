class ApplyZScore(Enum):
    ALL = 'all'
    PARTICIPANT = 'participant'
    TRIAL = 'trial'

    def __str__(self) -> str:
        return self.value

class Method(Enum):
    PCA = 'pca'
    MCCA = 'mcca'

    def __str__(self) -> str:
        return self.value

def user_input_n_comp(data):

    n_comp = np.shape(data)[0] - 1
    fig, ax = plt.subplots(1, 2, figsize=(0.2 * n_comp, 4))
    pca = PCA(n_components=n_comp, svd_solver="full", copy=False)  # selecting PCs
    pca.fit(data)

    ax[0].plot(np.arange(pca.n_components) + 1, pca.explained_variance_ratio_, ".-")
    ax[0].set_ylabel("Normalized explained variance")
    ax[0].set_xlabel("Component")
    ax[1].plot(np.arange(pca.n_components) + 1, np.cumsum(pca.explained_variance_ratio_), ".-")
    ax[1].set_ylabel("Cumulative normalized explained variance")
    ax[1].set_xlabel("Component")
    plt.tight_layout()
    plt.show()

    #TODO: needs user input validation?
    n_comp = int(
        input(
            f"How many PCs (95 and 99% explained variance at component "
            f"n{np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.95)[0][0] + 1} and "
            f"n{np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.99)[0][0] + 1}; "
            f"components till n{np.where(pca.explained_variance_ratio_ >= 0.01)[0][-1] + 1} "
            f"explain at least 1%)?"
        )
    )

    return n_comp

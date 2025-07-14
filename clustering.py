import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from util import RunningNorm


class SmartClusteringNovelty:
    def __init__(self, initial_clusters=80, adaptation_frequency=600_000):
        self.n_clusters = initial_clusters
        self.adaptation_frequency = adaptation_frequency
        self.state_buffer = []
        self.steps = 0
        self.buffer_size = 400_000
        self.batch_size = 10000  # for MiniBatchKMeans
        # Initialize model using MiniBatchKMeans
        self.model = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.batch_size)
        self.is_fitted = False
        self.reward_scale = 1
        self.running_norm = RunningNorm()
        self.max_clusters = 100
        self.min_clusters = 4

    def update(self, states_batch):
        """
        Update clustering model with a batch of observed states.
        Args:
            states_batch (numpy.ndarray or torch.Tensor): shape (batch_size, state_dim)
        """
        # Convert to numpy array if necessary
        if not isinstance(states_batch, np.ndarray):
            states_batch = states_batch.cpu().detach().numpy()
        states_batch = states_batch.reshape(states_batch.shape[0], -1)

        # Update step counter
        self.steps += len(states_batch)

        # Extend state buffer and ensure it doesn’t exceed buffer_size
        self.state_buffer.extend(states_batch)
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer = self.state_buffer[-self.buffer_size:]

        # Initial fit if not yet fitted
        if not self.is_fitted and len(self.state_buffer) >= self.n_clusters * 2:
            buffer_array = np.array(self.state_buffer)
            self.model.fit(buffer_array)
            self.is_fitted = True
            print("[SmartClusteringNovelty] Initial fitting done.")
            return

        # Regular incremental update
        if self.is_fitted:
            self.model.partial_fit(states_batch)

        # Periodically adjust the cluster count using the silhouette method
        # This check ensures adaptation runs only once per adaptation_frequency steps.
        if self.steps % self.adaptation_frequency < len(states_batch):
            self.adapt_cluster_count()
            self.steps = 0

    def adapt_cluster_count(self):
        """
        Adapt the number of clusters using silhouette analysis.
        """
        if len(self.state_buffer) < self.n_clusters * 2:
            return

        buffer_array = np.array(self.state_buffer)
        # For speed, sample up to 1000 states randomly from the buffer
        sample_size = min(10000, len(buffer_array))
        sample_indices = np.random.choice(len(buffer_array), sample_size, replace=False)
        sample_array = buffer_array[sample_indices]

        # Get current labels from existing model on the sample
        current_labels = self.model.predict(sample_array)
        # In order to compute silhouette score, we need at least 2 distinct clusters.
        if len(np.unique(current_labels)) < 2:
            current_score = -1  # low score if not enough clusters
        else:
            # current_score = silhouette_score(sample_array, current_labels, metric='euclidean')
            current_score = calinski_harabasz_score(sample_array, current_labels)

        # Now explore candidate cluster counts.
        best_k = self.n_clusters
        best_score = current_score

        # Candidates from min_clusters to max_clusters (you can adjust the step if desired)
        for candidate_k in range(self.min_clusters, self.max_clusters + 1):
            # Skip current candidate if equal to current cluster count.
            if candidate_k == self.n_clusters:
                continue
            # Fit a temporary model on the sample
            temp_model = MiniBatchKMeans(n_clusters=candidate_k, batch_size=self.batch_size)
            temp_model.fit(sample_array)
            candidate_labels = temp_model.predict(sample_array)
            # Only compute silhouette if at least two clusters are assigned
            if len(np.unique(candidate_labels)) < 2:
                continue
            # candidate_score = silhouette_score(sample_array, candidate_labels, metric='euclidean')
            candidate_score = calinski_harabasz_score(sample_array, candidate_labels)
            if candidate_score > best_score:
                best_score = candidate_score
                best_k = candidate_k

        # If best candidate differs from current, update the model.
        if best_k != self.n_clusters:
            if best_k < self.n_clusters:
                best_k = self.n_clusters - 1
            else:
                best_k = self.n_clusters + 1
            print(f"[SmartClusteringNovelty] Adapting clusters via silhouette: {self.n_clusters} → {best_k} (score: {best_score:.3f})")
            self.n_clusters = best_k
            self.model = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.batch_size)
            self.model.fit(buffer_array)
        else:
            print(f"[SmartClusteringNovelty] Keeping cluster count at {self.n_clusters} (silhouette score: {best_score:.3f})")

    def recluster(self):
        """
        Re-cluster using the entire state buffer when needed.
        """
        print(f"Reclustering at step {self.steps}")
        buffer_array = np.array(self.state_buffer)
        if len(buffer_array) >= self.n_clusters:
            self.model = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.batch_size)
            self.model.fit(buffer_array)
            self.is_fitted = True

    def __call__(self, states):
        return self.compute_novelty_reward(states)

    def compute_novelty_reward(self, states):
        """
        Compute intrinsic novelty reward for a batch of states.
        Args:
            states (numpy.ndarray or torch.Tensor): shape (n, state_dimension)
        Returns:
            rewards (numpy.ndarray): shape (n,), intrinsic rewards for each state.
        """
        if not self.is_fitted:
            # If the model is not ready, return small default rewards.
            return np.full(states.shape[0], 0.1 * self.reward_scale)

        # Convert input to numpy array if needed
        if not isinstance(states, np.ndarray):
            states = states.detach().cpu().numpy()
        states = states.reshape(states.shape[0], -1)

        # Compute distances from each state to each cluster center.
        cluster_centers = self.model.cluster_centers_
        distances = np.linalg.norm(
            states[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :],
            axis=-1
        )  # shape (n, num_clusters)

        # The novelty reward is based on the minimum distance.
        min_distances = np.min(distances, axis=1)

        # Normalize the novelty reward using the running normalization.
        normalized_rewards = self.running_norm(min_distances)
        clipped_result = np.clip(normalized_rewards, -3.0, 3.0)
        return clipped_result * self.reward_scale


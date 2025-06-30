import torch
import numpy as np
from typing import List, Dict
from client.compression import compress_gradient, decompress_gradient
from client.trainer import local_train, get_model_gradient
from client.privacy import clip_and_fake
from client.sampling import sample_uniform_data
from server.stratification import (
    stratify_clients, compute_stratum_statistics,
    neyman_allocation, importance_sample
)
from server.privacy import estimate_total_sample_size
from server.aggregation import aggregate_models

class FedSTaSCoordinator:
    def __init__(
            self,
            global_model: torch.nn.Module,
            client_datasets: List[torch.utils.data.Dataset],
            config: Dict,
            device: str = "cpu",
            verbose: bool = True
    ):
        self.global_model = global_model
        self.client_datasets = client_datasets
        self.device = device
        self.config = config
        self.num_clients = len(client_datasets)
        self.verbose = verbose

    def run(self, num_rounds: int):
        for round_idx in range(num_rounds):
            print(f"\n=== Round {round_idx + 1} ===")

            # Step 1: Compress gradients (IS)
            compressed_grads = []
            for i, dataset in enumerate(self.client_datasets):
                dummy_model = self.global_model.to(self.device)
                updated_model = local_train(
                    model = dummy_model,
                    dataset=dataset,
                    epochs=1,
                    batch_size=self.config["batch_size"],
                    lr=self.config["lr"],
                    sample_fraction=1.0, # Full data to simulate gradient
                    device=self.device
                )
                grad = get_model_gradient(dummy_model, updated_model).cpu().numpy()
                centroids, indices = compress_gradient(grad, self.config["d_prime"])
                compressed_grads.append((centroids, indices))
            
            # Step 2: Reconstruct gradients for stratification
            reconstructed = [
                decompress_gradient(c, i) for (c, i) in compressed_grads
            ]

            # Step 3: Stratify clients
            strata = stratify_clients(reconstructed, self.config["H"])
            S_h = compute_stratum_statistics(reconstructed, strata)
            N_h = {h: len(clients) for h, clients in strata.items()}
            m_h = neyman_allocation(N_h, S_h, self.config["clients_per_round"])
            
            if self.verbose:
                print("\n[Stratification]")
                for h, clients in strata.items():
                    print(f"  Stratum {h}: N_h = {len(clients)}, S_h = {S_h[h]:.4f}, m_h = {m_h[h]}")

            # Step 4: Sample clients via importance sampling
            selected_clients_by_stratum = {}
            if self.verbose:
                print("\n[Client Selection]")
            for h, client_indices in strata.items():
                if not client_indices or m_h[h] == 0:
                    continue
                norms = [np.linalg.norm(reconstructed[k]) for k in client_indices]
                selected = importance_sample(client_indices, norms, m_h[h])
                selected_clients_by_stratum[h] = selected
                if self.verbose:
                    print(f"  Stratum {h}: selected {selected}")
            
            # Step 5: Collect privatized sample counts
            responses = []
            if self.verbose:
                print("\n[Sample Size Reporting]")
            for h, clients in selected_clients_by_stratum.items():
                for k in clients:
                    n_k = len(self.client_datasets[k])
                    r_k = clip_and_fake(n_k, self.config["M"], self.config["alpha"])
                    responses.append(r_k)
                    if self.verbose:
                        print(f"  Client {k}: n_k = {n_k}, r_k = {r_k}")
            
            # Step 6: Estimate total sample count
            n_tilde = estimate_total_sample_size(responses, self.config["alpha"], self.config["M"])
            p = min(self.config["n_star"] / n_tilde, 1.0)
            print(f"Estimated ñ = {n_tilde:.2f}, using sampling ratio p = {p:.4f}")

            # Step 7: Train selected clients
            models_by_stratum = {}
            if self.verbose:
                print("\n[Local Training]")
            for h, clients in selected_clients_by_stratum.items():
                local_models = []
                for k in clients:
                    model_copy = self.global_model.to(self.device)
                    subset = sample_uniform_data(
                        self.client_datasets[k], p, seed=round_idx
                    )
                    if self.verbose:
                        print(f"  Client {k}: training on {len(subset)} samples")
                    updated_model = local_train(
                        model=model_copy,
                        dataset=subset,
                        epochs=self.config["epochs"],
                        batch_size=self.config["batch_size"],
                        lr=self.config["lr"],
                        sample_fraction=1.0,
                        device=self.device
                    )
                    local_models.append(updated_model)
                models_by_stratum[h] = local_models
            
            # Step 8: Aggregate updates
            self.global_model = aggregate_models(models_by_stratum, N_h, m_h)
            print("Aggregated global model updated.")

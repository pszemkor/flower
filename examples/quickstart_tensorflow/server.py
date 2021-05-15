import flwr as fl

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights
    
# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    strategy = SaveModelStrategy(
        fraction_fit=1.0,  # Sample 10% of available clients for the next round
        min_fit_clients=2,  # Minimum number of clients to be sampled for the next round
        min_available_clients=2)  # Minimum number of clients that need to be connected to the server before a training round can start)
    
    fl.server.start_server("0.0.0.0:8080", config={"num_rounds": 30}, strategy=strategy)

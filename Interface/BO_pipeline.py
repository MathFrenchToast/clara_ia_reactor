import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from typing import Optional
from scipy.spatial.distance import cdist

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False

# BoTorch & GPyTorch imports
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. INITIALIZATION DATA CLASS
# ==========================================
class BOInitData:
    """
    Sélection de points initiaux pour Bayesian Optimization.
    """
    def __init__(self,
                 n: int = 15,
                 method: str = 'max_min_dist',
                 metric: str = 'euclidean',
                 cluster_init: str = 'k-means++',
                 seed: int = 42):

        self.n = n
        self.method = method
        self.metric = metric
        self.cluster_init = cluster_init
        self.seed = seed

        self.method_map = {
            'kmedoids': self.kmedoids,
            'kmeans': self.kmeans,
            'max_min_dist': self.max_min_dist,
            'random': self.random,
        }

    def fit(self, X):
        """X peut être DataFrame ou numpy array"""
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float64)
        else:
            X_np = np.asarray(X, dtype=np.float64)

        if self.method != 'random':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_np)
        else:
            X_scaled = X_np

        init_method = self.method_map.get(self.method)
        if init_method is None:
            raise ValueError(f"Méthode inconnue: {self.method}")

        return init_method(X_scaled)

    def kmedoids(self, X_scaled):
        if not HAS_KMEDOIDS:
            raise ImportError("The 'scikit-learn-extra' package is required to use the 'kmedoids' method. "
                              "Please install it or use another method (e.g., 'max_min_dist' or 'kmeans').")
        kmedoids = KMedoids(
            n_clusters=self.n,
            init=self.cluster_init,
            random_state=self.seed,
            metric=self.metric,
            max_iter=5000,
        ).fit(X_scaled)
        return torch.tensor(kmedoids.medoid_indices_.tolist())

    def kmeans(self, X_scaled):
        kmeans = KMeans(
            n_clusters=self.n,
            init=self.cluster_init,
            random_state=self.seed,
            max_iter=5000,
        ).fit(X_scaled)
        centroids = torch.tensor(kmeans.cluster_centers_)
        X_torch = torch.from_numpy(X_scaled)
        distances = torch.norm(X_torch.unsqueeze(1) - centroids, dim=2)
        return torch.argmin(distances, dim=0)

    def max_min_dist(self, X_scaled):
        n_samples = X_scaled.shape[0]
        np.random.seed(self.seed)
        selected = [np.random.randint(0, n_samples)]

        for _ in range(self.n - 1):
            dist_to_selected = cdist(X_scaled, X_scaled[selected], metric=self.metric)
            min_dist = dist_to_selected.min(axis=1)
            next_idx = min_dist.argmax()
            selected.append(next_idx)
        return torch.tensor(selected)

    def random(self, X_scaled):
        np.random.seed(self.seed)
        indices = np.random.permutation(len(X_scaled))[:self.n]
        return torch.tensor(indices)
    

# ==========================================
# 2. DATA MODULE CLASS
# ==========================================
class BODataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "Interface/data/v2c_featurized_ml_ready.csv",
        n_init: int = 15,
        init_method: str = 'max_min_dist',
        target: str = "both",      # "both", "CH4", ou "CO2"
        max_feed: Optional[float] = None,
        max_temp: Optional[float] = None,
        max_time: Optional[float] = None,
        max_loading: Optional[float] = None,
        max_pore_size: Optional[float] = None,
        max_pore_volume: Optional[float] = None,
        max_surface_area: Optional[float] = None,
        max_H2_TPR_peak_temp: Optional[float] = None,
        max_particle_size: Optional[float] = None,
        max_ghsv: Optional[float] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.data_path = data_path
        self.n_init = n_init
        self.init_method = init_method
        self.target = target
        self.max_feed = max_feed
        self.max_temp = max_temp
        self.max_loading = max_loading
        self.max_time = max_time
        self.max_pore_size = max_pore_size
        self.max_pore_volume = max_pore_volume
        self.max_surface_area = max_surface_area
        self.max_H2_TPR_peak_temp = max_H2_TPR_peak_temp
        self.max_particle_size = max_particle_size
        self.max_ghsv = max_ghsv
        self.seed = seed

        self.init_data = BOInitData(n=n_init, method=init_method, seed=seed)
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        df = pd.read_csv(self.data_path)

        # Filter by Ratio of CH4 in Feed constraint if provided
        if self.max_feed is not None:
            df = df[df['Ratio of CH4 in Feed'] <= self.max_feed]

        # Filter by temperature constraint if provided
        if self.max_temp is not None:
            df = df[df['Reaction Temperature'] <= self.max_temp]

        # Filter by Ni loading constraint if provided
        if self.max_loading is not None:
            df = df[df['Ni Loading'] <= self.max_loading]

        # Filter by reaction time constraint if provided
        if self.max_time is not None:
            df = df[df['Reaction Time'] <= self.max_time]

        # Filter by pore size constraint if provided
        if self.max_pore_size is not None:
            df = df[df['Pore Size'] <= self.max_pore_size]

        # Filter by pore volume constraint if provided
        if self.max_pore_volume is not None:
            df = df[df['Pore Volume'] <= self.max_pore_volume]
        
        # Filter by surface area constraint if provided
        if self.max_surface_area is not None:
            df = df[df['Surface Area'] <= self.max_surface_area]

        # Filter by H2-TPR Peak Temperature constraint if provided
        if self.max_H2_TPR_peak_temp is not None:
            df = df[df['H2-TPR Peak Temperature'] <= self.max_H2_TPR_peak_temp]

        # Filter by Ni particle size constraint if provided
        if self.max_particle_size is not None:
            df = df[df['Ni Particle Size'] <= self.max_particle_size]

        # Filter by GHSV constraint if provided
        if self.max_ghsv is not None:
            df = df[df['GHSV'] <= self.max_ghsv]

        feature_cols = [
            'Ratio of CH4 in Feed', 'Reaction Temperature', 'Ni Loading',
            'Reaction Time', 'Pore Size', 'Pore Volume', 'Surface Area',
            'H2-TPR Peak Temperature', 'Ni Particle Size', 'GHSV', 'PC1', 
            'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'
        ]

        X = df[feature_cols].values.astype(np.float64)

        if self.target == "both":
            y = df[['CH4 Conversion', 'CO2 Conversion']].values.astype(np.float64)
        elif self.target == "CH4":
            y = df['CH4 Conversion'].values.reshape(-1, 1).astype(np.float64)
        else:
            y = df['CO2 Conversion'].values.reshape(-1, 1).astype(np.float64)

        # Filtre NaN
        valid_mask = ~np.isnan(y).any(axis=1)
        self.X = X[valid_mask]
        self.y = y[valid_mask]
        self.df_valid = df[valid_mask].reset_index(drop=True)

        # Sélection des points initiaux
        init_indices = self.init_data.fit(self.X)

        # Split
        self.train_x = torch.from_numpy(self.X[init_indices.numpy()]).to(torch.float64)
        self.train_y = torch.from_numpy(self.y[init_indices.numpy()]).to(torch.float64)

        heldout_mask = torch.ones(len(self.X), dtype=torch.bool)
        heldout_mask[init_indices] = False

        self.heldout_x = torch.from_numpy(self.X[heldout_mask]).to(torch.float64)
        self.heldout_y = torch.from_numpy(self.y[heldout_mask]).to(torch.float64)

    def train_dataloader(self):
        dataset = TensorDataset(self.train_x, self.train_y)
        return DataLoader(dataset, batch_size=len(self.train_x), shuffle=False)

    def heldout_dataloader(self):
        dataset = TensorDataset(self.heldout_x, self.heldout_y)
        return DataLoader(dataset, batch_size=len(self.heldout_x), shuffle=False)

# ==========================================
# 3. GAUSSIAN PROCESS CLASS
# ==========================================
class GP(SingleTaskGP):
    def __init__(self, train_x, train_y):
        # Ensure train_y is at least 2-dimensional (batch_shape, num_outcomes)
        if train_y.dim() == 1:
            train_y = train_y.unsqueeze(-1)

        outcome_transform = Standardize(m=train_y.shape[-1])
        input_transform = Normalize(d=train_x.shape[-1])

        super().__init__(
            train_X=train_x,
            train_Y=train_y,
            outcome_transform=outcome_transform,
            input_transform=input_transform
        )
        self.covar_module = ScaleKernel(MaternKernel(ard_num_dims=train_x.shape[-1]))

    def reinit(self, train_x, train_y):
        self.__init__(train_x, train_y)

# ==========================================
# 4. MAIN OPTIMIZATION LOOP
# ==========================================
def run_bo_continuous(
    dm,
    bounds,
    feature_names,
    fixed_params=None,
    num_iterations=30,
):
    train_x = dm.train_x.clone()
    train_y = dm.train_y.clone()

    hypervolume_history = []
    best_ch4_history = []
    best_co2_history = []
    
    new_x = None

    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}")

        # 1. GP MULTI-OUTPUT
        gp_ch4 = GP(train_x, train_y[:, [0]])
        gp_co2 = GP(train_x, train_y[:, [1]])
        model = ModelListGP(gp_ch4, gp_co2).double()

        mll = SumMarginalLogLikelihood(model.likelihood, model)
        
        import warnings
        from botorch.exceptions.warnings import OptimizationWarning
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                with gpytorch.settings.cholesky_jitter(1e-3):
                    fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"GP fitting stopped early at iteration {i+1} due to numerical instability: {e}")
            break

        # 2. POINT DE REFERENCE
        ref_point = train_y.min(dim=0).values - 0.01
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_y)

        # 3. ACQUISITION FUNCTION
        acq_func = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
        )

        # 4. CONTRAINTES
        bounds_opt = bounds.clone()
        if fixed_params is not None:
            for key, value in fixed_params.items():
                if key in feature_names:
                    idx = feature_names.index(key)
                    bounds_opt[0, idx] = value
                    bounds_opt[1, idx] = value

        # 5. OPTIMISATION
        candidate, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds_opt,
            q=1,
            num_restarts=15,
            raw_samples=100,
        )

        new_x = candidate.detach()

        # 6. NEAREST NEIGHBOR
        X_tensor = torch.tensor(dm.X, dtype=torch.float64)
        
        # Scale features using bounds for distance calculation to prevent large-scale features from dominating
        scale_range = bounds_opt[1] - bounds_opt[0]
        scale_range = torch.where(scale_range == 0, torch.ones_like(scale_range), scale_range)
        
        X_scaled = (X_tensor - bounds_opt[0]) / scale_range
        new_x_scaled = (new_x - bounds_opt[0]) / scale_range
        
        distances = torch.norm(X_scaled - new_x_scaled, dim=1)
        
        # Mask out points already in train_x to avoid selecting them again
        dist_to_train = torch.cdist(X_tensor, train_x)
        already_evaluated = dist_to_train.min(dim=1).values < 1e-4
        distances[already_evaluated] = float('inf')
        
        closest_idx = distances.argmin()

        if distances[closest_idx] == float('inf'):
            print(f"Convergence reached at iteration {i+1}. All dataset points evaluated.")
            break
            
        actual_x = X_tensor[closest_idx].unsqueeze(0)
        new_y = torch.tensor(dm.y[closest_idx], dtype=torch.float64).unsqueeze(0)

        # 7. UPDATE DATASET (Using actual dataset points, not candidate points)
        train_x = torch.cat([train_x, actual_x])
        train_y = torch.cat([train_y, new_y])
        new_x = actual_x  # Update new_x so the returned suggestion matches the actual point

        # 8. CALCUL HYPERVOLUME
        pareto_mask = is_non_dominated(train_y)
        pareto_y = train_y[pareto_mask]

        hv = Hypervolume(ref_point)
        hv_value = hv.compute(pareto_y)
        hypervolume_history.append(hv_value)

        # 9. SUIVI DES SCORES
        best_ch4 = train_y[:, 0].max().item()
        best_co2 = train_y[:, 1].max().item()

        best_ch4_history.append(best_ch4)
        best_co2_history.append(best_co2)
        
        print(f"--> Current Best CH4: {best_ch4:.2f}%, Current Best CO2: {best_co2:.2f}%")
        
    # Format the final suggested parameters for the Streamlit UI mapping
    next_best_params = {}
    if new_x is not None:
        next_best_params = {name: val.item() for name, val in zip(feature_names, new_x.squeeze())}

    return (
        train_x,
        train_y,
        np.array(hypervolume_history),
        np.array(best_ch4_history),
        np.array(best_co2_history),
        next_best_params
    )

# Alias for backward compatibility or user preference
run_bo_optimization = run_bo_continuous

if __name__ == "__main__":
    # Test script for the BO pipeline
    DATA_PATH = "Interface/data/v2c_featurized_ml_ready.csv"
    MAX_TEMP = 700.0  # Constraint for the reactor
    
    # Initialize Data Module
    print(f"Initializing Data Module (Max Temp: {MAX_TEMP})...")
    dm = BODataModule(
        data_path=DATA_PATH,
        n_init=15,
        init_method='max_min_dist',
        target="both",
        max_temp=MAX_TEMP,
        seed=42
    )
    
    # Show initial best values
    initial_best_ch4 = dm.train_y[:, 0].max().item()
    initial_best_co2 = dm.train_y[:, 1].max().item()
    print(f"Initial best CH4 Conversion: {initial_best_ch4:.2f}%")
    print(f"Initial best CO2 Conversion: {initial_best_co2:.2f}%")

    print("\nInitial 15 points (Train Set):")
    # Feature columns as defined in BODataModule.setup
    feature_cols = [
        'Ratio of CH4 in Feed', 'Reaction Temperature', 'Ni Loading',
        'Reaction Time', 'Pore Size', 'Pore Volume', 'Surface Area',
        'H2-TPR Peak Temperature', 'Ni Particle Size', 'GHSV',
        'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'
    ]
    
    train_df = pd.DataFrame(dm.train_x.numpy(), columns=feature_cols)
    train_df['CH4 Conv'] = dm.train_y[:, 0].numpy()
    train_df['CO2 Conv'] = dm.train_y[:, 1].numpy()
    print(train_df.to_string(index=False))
    
    # Define bounds from the dataset
    df = pd.read_csv(DATA_PATH)
    # Apply the same filter for bound calculation
    df_filtered = df[df['Reaction Temperature'] <= MAX_TEMP]
    X_data = df_filtered[feature_cols].values
    
    min_bounds = torch.tensor(np.nanmin(X_data, axis=0), dtype=torch.float64)
    max_bounds = torch.tensor(np.nanmax(X_data, axis=0), dtype=torch.float64)
    
    # Explicitly ensure the temperature bound in the search space is capped
    temp_idx = feature_cols.index('Reaction Temperature')
    max_bounds[temp_idx] = MAX_TEMP
    
    bounds = torch.stack([min_bounds, max_bounds])
    
    # Run a short optimization for testing
    print("Starting Bayesian Optimization test...")
    results = run_bo_continuous(
        dm=dm,
        bounds=bounds,
        feature_names=feature_cols,
        num_iterations=30  # Increased for better convergence in 20D space
    )
    
    train_x, train_y, hv_hist, ch4_hist, co2_hist, next_params = results
    
    print("\nTest completed successfully!")
    print(f"Final training set size: {train_x.shape[0]} points")
    print(f"Best CH4 Conversion found: {ch4_hist[-1]:.2f}%")
    print(f"Best CO2 Conversion found: {co2_hist[-1]:.2f}%")
    print(f"Hypervolume history: {hv_hist}")
    print(f"Suggested next experiment: {next_params}")

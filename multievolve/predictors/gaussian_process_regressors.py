from joblib import Parallel, delayed
from math import ceil

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import iqr
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    DotProduct as DP,
    RBF
)

from multievolve.predictors.base_regressors import BaseRegressor
from multievolve.utils.other_utils import performance_report

def parallel_predict(model, X, batch_num, n_batches, verbose):
    """
    Makes predictions in parallel using batches.

    Args:
        model: The trained model to make predictions with
        X (array-like): Input features to predict on
        batch_num (int): Current batch number
        n_batches (int): Total number of batches
        verbose (bool): Whether to print progress messages

    Returns:
        tuple: (mean predictions, prediction variances)
    """
    mean, var = model.predict(X, return_std=True)
    if verbose:
        print('Finished predicting batch number {}/{}'
               .format(batch_num + 1, n_batches))
    return mean, var

class GPRegressor(BaseRegressor):
    """Base Gaussian Process regressor class.

    Attributes:
        n_restarts_ (int): Number of restarts for optimizer
        kernel_ (sklearn.gaussian_process.kernels): Kernel function
        normalize_y_ (bool): Whether to normalize target values
        backend_ (str): Backend framework to use ('sklearn', 'gpy', or 'gpytorch')
        batch_size_ (int): Batch size for predictions
        n_jobs_ (int): Number of parallel jobs
        verbose_ (bool): Whether to print progress messages
        model: The trained GP model
        uncertainties_ (array): Prediction uncertainties
    """

    def __init__(self, 
                data_splitter, 
                featurizer,
                model='GPR',
                n_restarts=0,
                kernel=None,
                normalize_y=True,
                backend='sklearn',
                batch_size=1000,
                n_jobs=1,
                verbose=False,
                **kwargs
                ):
        """
        Args:
            data_splitter: Object to split data into train/test sets
            featurizer: Object to convert sequences to numerical features
            model (str): Model identifier string
            n_restarts (int): Number of restarts for optimizer
            kernel: Kernel function for GP
            normalize_y (bool): Whether to normalize target values
            backend (str): Framework to use ('sklearn', 'gpy', or 'gpytorch')
            batch_size (int): Batch size for predictions
            n_jobs (int): Number of parallel jobs
            verbose (bool): Whether to print progress messages
            **kwargs: Additional keyword arguments
        """
        self.n_restarts_ = n_restarts
        self.kernel_ = kernel
        self.normalize_y_ = normalize_y
        self.backend_ = backend
        self.batch_size_ = batch_size
        self.n_jobs_ = n_jobs
        self.verbose_ = verbose
        super().__init__(data_splitter, featurizer, model, **kwargs)
    
    def train(self, X, y):
        """
        Train the GP model.

        Args:
            X (array-like): Training features
            y (array-like): Training target values

        Returns:
            self: The trained model instance
        """
        n_samples, n_features = X.shape

        if self.verbose_:
            print('Fitting GP model on {} data points with dimension {}...'
                   .format(*X.shape))

        # scikit-learn backend.
        if self.backend_ == 'sklearn':
            from sklearn.gaussian_process import GaussianProcessRegressor
            self.model = GaussianProcessRegressor(
                kernel=self.kernel_,
                normalize_y=self.normalize_y_,
                alpha=1e-0,
                n_restarts_optimizer=self.n_restarts_,
                copy_X_train=False,
            ).fit(X, y)

        # GPy backend.
        elif self.backend_ == 'gpy':
            import GPy
            if self.kernel_ == 'rbf':
                kernel = GPy.kern.RBF(
                    input_dim=n_features, variance=1., lengthscale=1.
                )
            else:
                raise ValueError('Kernel value {} not supported'
                                 .format(self.kernel_))

            self.model = GPy.models.SparseGPRegression(
                X, y.reshape(-1, 1), kernel=kernel,
                num_inducing=min(self.n_inducing_, n_samples)
            )
            self.model.Z.unconstrain()
            self.model.optimize(messages=self.verbose_)

        # GPyTorch with CUDA backend.
        elif self.backend_ == 'gpytorch':
            import gpytorch
            import torch

            class GPyTorchRegressor(gpytorch.models.ExactGP):
                def __init__(self, X, y, likelihood):
                    super(GPyTorchRegressor, self).__init__(X, y, likelihood)
                    self.mean_module = gpytorch.means.ConstantMean()
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.RBFKernel()
                    )
            
                def forward(self, X):
                    mean_X = self.mean_module(X)
                    covar_X = self.covar_module(X)
                    return gpytorch.distributions.MultivariateNormal(mean_X, covar_X)
            
            X = torch.Tensor(X).contiguous().cuda()
            y = torch.Tensor(y).contiguous().cuda()

            likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            model = GPyTorchRegressor(X, y, likelihood).cuda()

            model.train()
            likelihood.train()

            # Use the Adam optimizer.
            #optimizer = torch.optim.LBFGS([ {'params': model.parameters()} ])
            optimizer = torch.optim.Adam([
                {'params': model.parameters()}, # Includes GaussianLikelihood parameters.
            ], lr=1.)

            # Loss for GPs is the marginal log likelihood.
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            training_iterations = 100
            for i in range(training_iterations):
                optimizer.zero_grad()
                output = model(X)
                loss = -mll(output, y)
                loss.backward()
                if self.verbose_:
                    print('Iter {}/{} - Loss: {:.3f}'
                           .format(i + 1, training_iterations, loss.item()))
                optimizer.step()

            self.model = model
            self.likelihood_ = likelihood

        if self.verbose_:
            print('Done fitting GP model.')

        return self

    def evaluate(self):
        """
        Evaluates the model on a test set.

        Returns:
            tuple: (dict of evaluation metrics, matplotlib figure)
        """
        
        # Reshape data and get correlation stats
        y_pred = self.predict(self.X_test)
        y, y_pred = np.array(self.y_test), np.array(y_pred)
        y, y_pred = y.reshape(-1), y_pred.reshape(-1)
        stats = performance_report(y, y_pred)

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 5))  # Adjust size as needed

        # Clip data points that have activity less than 0 or greater than 1.2x the max experimental y value
        y_max = y.max()*1.1
        y_pred_adjusted = np.clip(y_pred, 0, y_max)

        # Scale uncertainties using IQR and color data points based on uncertainty
        scaled_uncertainties = (self.uncertainties_ - np.percentile(self.uncertainties_, 25)) / iqr(self.uncertainties_)
        cmap = plt.cm.viridis
        colors = cmap(scaled_uncertainties)
        colors[y_pred > y_max] = mcolors.to_rgba('crimson')
        colors[y_pred < 0] = mcolors.to_rgba('crimson')

        # Scatter plot for main graph
        scatter = ax.scatter(y_pred_adjusted, y, c=colors, alpha = 0.4)

        # Uncertainty colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Uncertainty')

        # Draw x=y line
        ax.plot([0, y_max], [0, y_max], 'k--', linewidth=2)

        # Set labels and title for main graph
        ax.text(0.9, 0.1, f'Pearson r={stats["Pearson r"]:.2f}', fontsize=12, ha='right', va='bottom', transform=ax.transAxes)
        ax.set_xlabel('Predicted Score')
        ax.set_ylabel('True Score')
        ax.set_title(f'Model Performance')
        ax.set_xlim(0, y_max)

        # Display model parameters using legend
        model_params = self.name.split('|')  # Assuming '|' separates different parameters
        param_text = '\n'.join(model_params)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, param_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

        # Return the figure and axes object
        return stats, fig

    def custom_predictor(self, X):
        """
        Makes predictions using the trained GP model.

        Args:
            X (array-like): Features to predict on

        Returns:
            array: Mean predictions
        """
        if self.verbose_:
            print('Finding GP model predictions on {} data points...'
                   .format(X.shape[0]))

        if self.backend_ == 'sklearn':
            n_batches = int(ceil(float(X.shape[0]) / self.batch_size_))
            results = Parallel(n_jobs=self.n_jobs_)(#, max_nbytes=None)(
                delayed(parallel_predict)(
                    self.model,
                    X[batch_num*self.batch_size_:(batch_num+1)*self.batch_size_],
                    batch_num, n_batches, self.verbose_
                )
                for batch_num in range(n_batches)
            )
            mean = np.concatenate([ result[0] for result in results ])
            var = np.concatenate([ result[1] for result in results ])

        elif self.backend_ == 'gpy':
            mean, var = self.model.predict(X, full_cov=False)

        elif self.backend_ == 'gpytorch':
            import gpytorch
            import torch

            X = torch.Tensor(X).contiguous().cuda()

            # Set into eval mode.
            self.model.eval()
            self.likelihood_.eval()

            with torch.no_grad(), \
                 gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.max_root_decomposition_size(35):
                preds = self.model(X)

            mean = preds.mean.detach().cpu().numpy()
            var = preds.variance.detach().cpu().numpy()

        if self.verbose_:
            print('Done predicting with GP model.')

        self.uncertainties_ = var.flatten()
        return mean.flatten()

class SparseGPRegressor(BaseRegressor):
    """Sparse Gaussian Process regressor using inducing points.

    Attributes:
        n_inducing_ (int): Number of inducing points
        method_ (str): Method for selecting inducing points ('uniform' or 'geosketch')
        n_restarts_ (int): Number of restarts for optimizer
        kernel_ (sklearn.gaussian_process.kernels): Kernel function
        backend_ (str): Backend framework to use
        batch_size_ (int): Batch size for predictions
        n_jobs_ (int): Number of parallel jobs
        verbose_ (bool): Whether to print progress messages
        gpr_: The trained GP model
    """

    def __init__(
            self, 
            data_splitter, 
            featurizer,
            model='SparseGPRegressor', 
            n_inducing=1000,
            method='geoskech',
            n_restarts=0,
            kernel=None,
            backend='sklearn',
            batch_size=1000,
            n_jobs=1,
            verbose=False,
            **kwargs
    ):
        """
        Args:
            data_splitter: Object to split data into train/test sets
            featurizer: Object to convert sequences to numerical features
            model (str): Model identifier string
            n_inducing (int): Number of inducing points
            method (str): Method for selecting inducing points
            n_restarts (int): Number of restarts for optimizer
            kernel: Kernel function for GP
            backend (str): Framework to use
            batch_size (int): Batch size for predictions
            n_jobs (int): Number of parallel jobs
            verbose (bool): Whether to print progress messages
            **kwargs: Additional keyword arguments
        """
        self.n_inducing_ = n_inducing
        self.method_ = method
        self.n_restarts_ = n_restarts
        self.kernel_ = kernel
        self.backend_ = backend
        self.batch_size_ = batch_size
        self.n_jobs_ = n_jobs
        self.verbose_ = verbose
        super().__init__(data_splitter, featurizer,model, **kwargs)

    def train(self, X, y):
        """
        Train the sparse GP model.

        Args:
            X (array-like): Training features
            y (array-like): Training target values
        """
        X, y = self.X, self.y
        if X.shape[0] > self.n_inducing_:
            if self.method_ == 'uniform':
                uni_idx = np.random.choice(X.shape[0], self.n_inducing_,
                                           replace=False)
                X_sketch = X[uni_idx]
                y_sketch = y[uni_idx]

            elif self.method_ == 'geosketch':
                from fbpca import pca
                from geosketch import gs

                U, s, _ = pca(X, k=100)
                X_dimred = U[:, :100] * s[:100]
                gs_idx = gs(X_dimred, self.n_inducing_, replace=False)
                X_sketch = X[gs_idx]
                y_sketch = y[gs_idx]

        else:
            X_sketch, y_sketch = X, y

        self.gpr_ = GPRegressor(
            n_restarts=self.n_restarts_,
            kernel=self.kernel_,
            backend=self.backend_,
            batch_size=self.batch_size_,
            n_jobs=self.n_jobs_,
            verbose=self.verbose_,
        ).fit(X_sketch, y_sketch)


    def custom_predictor(self, X):
        """
        Makes predictions using the trained sparse GP model.

        Args:
            X (array-like): Features to predict on

        Returns:
            array: Mean predictions
        """
        y_pred = self.gpr_.predict(X)
        self.uncertainties_ = self.gpr_.uncertainties_
        return y_pred

class GPLinearRegressor(GPRegressor):
    """Gaussian Process regressor with linear kernel."""

    def __init__(self, 
                 data_splitter, 
                 featurizer,
                 model='GPLinearRegressor',
                 n_restarts=0,
                 kernel = C(1., 'fixed') * DP(1., 'fixed'),
                 normalize_y=True,
                 backend='sklearn',
                 batch_size=1000,
                 n_jobs=1,
                 verbose=False,
                 **kwargs
                 ):
        """
        Args:
            data_splitter: Object to split data into train/test sets
            featurizer: Object to convert sequences to numerical features
            model (str): Model identifier string
            n_restarts (int): Number of restarts for optimizer
            kernel: Linear kernel function
            normalize_y (bool): Whether to normalize target values
            backend (str): Framework to use
            batch_size (int): Batch size for predictions
            n_jobs (int): Number of parallel jobs
            verbose (bool): Whether to print progress messages
            **kwargs: Additional keyword arguments
        """
        
        super().__init__(data_splitter, featurizer, model, n_restarts, kernel, normalize_y, backend, batch_size, n_jobs, verbose, **kwargs)

class GPQuadRegressor(GPRegressor):
    """Gaussian Process regressor with quadratic kernel."""

    def __init__(self,
                data_splitter, 
                featurizer,
                model='GPQuadRegressor',
                n_restarts=0,
                kernel = C(1., 'fixed') * (DP(1, 'fixed') ** 2),
                normalize_y=True,
                backend='sklearn',
                batch_size=1000,
                n_jobs=1,
                verbose=False,
                **kwargs
                ):
        """
        Args:
            data_splitter: Object to split data into train/test sets
            featurizer: Object to convert sequences to numerical features
            model (str): Model identifier string
            n_restarts (int): Number of restarts for optimizer
            kernel: Quadratic kernel function
            normalize_y (bool): Whether to normalize target values
            backend (str): Framework to use
            batch_size (int): Batch size for predictions
            n_jobs (int): Number of parallel jobs
            verbose (bool): Whether to print progress messages
            **kwargs: Additional keyword arguments
        """
    
        super().__init__(data_splitter, featurizer, model, n_restarts, kernel, normalize_y, backend, batch_size, n_jobs, verbose, **kwargs)

class GPRBFRegressor(GPRegressor):
    """Gaussian Process regressor with RBF kernel."""

    def __init__(self, 
                 data_splitter, 
                 featurizer,
                model='GPRBFRegressor',
                n_restarts=0,
                kernel = C(1., 'fixed') * RBF(1., 'fixed'),
                normalize_y=True,
                backend='sklearn',
                batch_size=1000,
                n_jobs=1,
                verbose=False,
                **kwargs
                ):
        """
        Args:
            data_splitter: Object to split data into train/test sets
            featurizer: Object to convert sequences to numerical features
            model (str): Model identifier string
            n_restarts (int): Number of restarts for optimizer
            kernel: RBF kernel function
            normalize_y (bool): Whether to normalize target values
            backend (str): Framework to use
            batch_size (int): Batch size for predictions
            n_jobs (int): Number of parallel jobs
            verbose (bool): Whether to print progress messages
            **kwargs: Additional keyword arguments
        """
    
        super().__init__(data_splitter, featurizer, model, n_restarts, kernel, normalize_y, backend, batch_size, n_jobs, verbose, **kwargs)
import os
import pickle
from abc import ABC, abstractmethod

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.metrics import make_scorer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor as RFRegressor

from multievolve.utils.other_utils import performance_report

def run_model_experiments(splits, 
                          features, 
                          models, 
                          experiment_name,
                          use_cache=False,
                          show_plots=True):
    """
    Trains multiple models with various data splits and features, evaluates their performance,
    and compiles the results into a CSV file.

    Args:
        splits (list): A list of data splits to use for training
        features (list): A list of features to use for training
        models (list): A list of model instances to train
        experiment_name (str): Name of the experiment for saving results
        use_cache (bool, optional): Whether to use cached models. Defaults to False.
        show_plots (bool, optional): Whether to show matplotlib plots. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing evaluation results for all models

    Example Usage:

    run_model_experiments(splits, 
                          features, 
                          models, 
                          experiment_name,
                          use_cache=False)
    """

    names = []
    stats = []

    for split in splits:
        for feature in features:
            for model in models:

                instance = model(split, feature, use_cache=use_cache, show_plots=show_plots)
                
                # Train and evaluate model
                stat = instance.run_model()
                names.append(instance.file_attrs['model_name'].split('__'))
                stats.append(list(stat.values()))

    # Return results for all training permutations
    stats_array = np.array(stats)
    names_array = np.array(names)
    combined_array = np.concatenate([names_array, stats_array], axis=1)
    columns = ['Data Split', 'Feature', 'Model'] + list(stat.keys())
    table = pd.DataFrame(combined_array, columns=columns)

    # Check if the directory exists, create it if it doesn't
    dir_path = f'{instance.file_attrs["model_dir"]}/' + "results"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Save the results
    if use_cache:
        table.to_csv(f"{dir_path}/{experiment_name}.csv", index=False)

    return table


class BaseRegressor(ABC):
    """
    Abstract base class for regression models.

    Args:
        data_splitter: Object containing train/test splits
        featurizer: Object that converts sequences to features
        model (str, optional): Name of model. Defaults to 'Base'
        use_cache (bool, optional): Whether to use cached models. Defaults to False
        show_plots (bool, optional): Whether to show matplotlib plots. Defaults to True
        **kwargs: Additional keyword arguments

    Attributes:
        model_name (str): Name of the model
        featurizer: Featurizer object
        use_cache (bool): Whether to use cached models
        kwargs (dict): Additional keyword arguments
        X_train: Training features
        X_test: Test features  
        y_train: Training labels
        y_test: Test labels
        split_method (str): Name of data split method
        file_attrs (dict): Dictionary of file attributes and paths
        show_plots (bool): Whether to show matplotlib plots. Defaults to True
    
    Example Usage:

    regressor = BaseRegressor(data_splitter, featurizer, model='Linear', use_cache=False, show_plots=True)
    regressor.run_model()
    """

    def __init__(self, 
                 data_splitter, 
                 featurizer, 
                 model='Base', 
                 use_cache=False, 
                 show_plots=True,
                 **kwargs):
        
        # Set variables
        self.model_name = model
        self.featurizer = featurizer
        self.use_cache = use_cache
        self.show_plots = show_plots
        self.kwargs = kwargs

        # Setup data
        self.X_train = data_splitter.splits['X_train']
        self.X_test = data_splitter.splits['X_test']
        self.y_train = data_splitter.splits['y_train']
        self.y_test = data_splitter.splits['y_test']
        self.split_method = data_splitter.splits['split_name']

        # Check if 'X_val' is not a key in data_splitter
        if 'X_val' in data_splitter.splits:
            print("Validation sets do not need to be present in data splits for non-neural network models.")

        # Set model directory
        self.file_attrs = data_splitter.file_attrs
        self.file_attrs['model_name'] = self.split_method + ' __ ' + self.featurizer.name + ' __ ' + self.model_name
        self.file_attrs['model_dir'] = os.path.join(data_splitter.file_attrs["dataset_dir"], 'model_cache', data_splitter.file_attrs["dataset_name"])
        self.file_attrs['model_path'] = os.path.join(self.file_attrs['model_dir'], 'objects', f'{self.file_attrs["model_name"]}.pkl')
        
        
        # Load model if available
        if self.file_attrs['model_path'] is not None and os.path.exists(self.file_attrs['model_path']) and self.use_cache:
            self.load_model(self.file_attrs['model_path'])

    def run_model(self, eval=True):
        """
        Runs model training and evaluation.

        Args:
            eval (bool, optional): Whether to evaluate the model. Defaults to True.

        Returns:
            If eval=True:
                dict: Dictionary of evaluation statistics
            If eval=False:
                None
        """

        if self.file_attrs['model_path'] is not None and os.path.exists(self.file_attrs['model_path']) and self.use_cache:
            pass
        else:
            print(f'Training model for {self.file_attrs["model_name"]}')
            X = self.preprocess_data(self.X_train)
            self.train(X, self.y_train)
            
            if self.use_cache:
                self.save_model()
        
        if eval:
            return self.evaluate()
        else:
            return None

            
    def load_model(self, model_path=None):
        """
        Loads a pre-trained model from a pkl file.

        Args:
            model_path (str, optional): Path to model file. Defaults to None.
        """

        # set location to load model
        model_path = self.file_attrs['model_path'] if model_path is None else model_path
        print(f'Loading model from {model_path}')

        try:
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print(f"Error: The file at {model_path} was not found.")
        except PermissionError:
            print(f"Error: Permission denied when trying to read {model_path}.")
        except Exception as e:
            print(f"An error occurred while loading the model: {str(e)}")

    def save_model(self, model_path=None):
        """
        Saves the model to a pkl file.

        Args:
            model_path (str, optional): Path to save model to. Defaults to None.
        """
        
        # set location to save model
        model_path = self.file_attrs['model_path'] if model_path is None else model_path

        dir_path = os.path.join(self.file_attrs['model_dir'], 'objects')
        # Check if the directory exists, create it if it doesn't
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Save the model
        print(f'Saving model to {model_path}')
        pickle.dump(self.model, open(model_path, 'wb'))

    def featurize(self, X):
        """
        Featurizes a list of sequences.

        Args:
            X (list): List of sequences to featurize

        Returns:
            array: Featurized sequences
        """

        X_featurized = self.featurizer.featurize(X)
        return X_featurized

    def preprocess_data(self, X):
        """
        Featurizes and scales input data.

        Args:
            X (list): List of sequences to preprocess

        Returns:
            array: Preprocessed data
        """
        X = self.featurizer.featurize(X)

        X = X.reshape(X.shape[0],-1)

        return X
        
    @abstractmethod
    def train(self, X, y):
        """
        Trains the model.

        Args:
            X (array): Input features
            y (array): Target values

        Returns:
            Trained model, also stored in self.model
        """
        pass

    def evaluate(self):
        """
        Evaluates model on test set.

        Returns:
            dict: Dictionary of evaluation statistics
        """
        
        # Evaluate model
        y_pred = self.predict(self.X_test)

        # Reshape data and get correlation stats
        y, y_pred = np.array(self.y_test), np.array(y_pred)
        y, y_pred = y.reshape(-1), y_pred.reshape(-1)

        # Get stats
        stats = performance_report(y, y_pred)

        # Set the default parameters
        plt.rcParams['font.size'] = 7
        plt.rcParams['lines.linewidth'] = 0.5

        # Plotting Results
        fig, ax = plt.subplots(figsize=(4, 3))  # Adjust size as needed
        
        ## Mark data points that have activity less than 0 or greater than 1.2x the max experimental y value
        y_max = max(y.max() * 1.2, y_pred.max() * 1.2)
        y_min = min(y.min() * 0.8, y_pred.min() * 0.8)
        # colors = np.where(y_pred > y_max, 'crimson', np.where(y_pred < 0, 'crimson', 'dodgerblue'))
        # y_pred_adjusted = np.clip(y_pred, 0, y_max)

        ## Scatter plot for main graph
        ax.scatter(y_pred, y, c='dodgerblue', alpha=0.4, edgecolors='w', linewidth=0.5)

        ## Draw x=y line
        ax.plot([y_min, y_max], [y_min, y_max], 'k--', linewidth=0.5)

        ## Set labels and title for main graph
        ax.text(0.9, 0.1, f'Pearson r={stats["Pearson r"]:.2f}', fontsize=7, ha='right', va='bottom', transform=ax.transAxes)
        ax.text(0.9, 0.2, f'Spearman r={stats["Spearman r"]:.2f}', fontsize=7, ha='right', va='bottom', transform=ax.transAxes)
        ax.set_xlabel('Predicted Score', fontsize=7)
        ax.set_ylabel('True Score', fontsize=7)
        ax.set_title('Model Performance', fontsize=7)
        ax.set_xlim(y_min, y_max)

        ## Display model parameters using legend
        model_params = self.file_attrs["model_name"].split('__')  # Assuming '|' separates different parameters
        param_text = '\n'.join(model_params)
        props = dict(boxstyle='square', facecolor='wheat', alpha=0.2)
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=7, verticalalignment='top', bbox=props)

        # Adjust tick parameters
        ax.tick_params(axis='both', which='major', labelsize=7)

        # Show figure
        if self.show_plots:
            plt.show()
        plt.close(fig)
        
        # Return the stats
        return stats

    @abstractmethod
    def custom_predictor(self, X):
        """
        Custom prediction method to be implemented in subclasses.
        Inputs have been filtered by self.predict()

        Args:
            X (array): Featurized sequences

        Returns:
            array: Model predictions
        """
        pass
        
    def predict(self, X):
        """
        Gets model predictions. Runs checks and calls custom_predictor.

        Args:
            X (list): List of sequences

        Returns:
            array: Model predictions
        """

        X_featurized = self.featurizer.featurize(X)

        X_featurized = X_featurized.reshape(X_featurized.shape[0],-1)

        predictions = self.custom_predictor(X_featurized)

        return predictions

class IdentityRegressor(BaseRegressor):
    """
    Identity regressor that returns all 1's.

    Args:
        data_splitter: Object containing train/test splits
        featurizer: Object that converts sequences to features
        model (str, optional): Name of model. Defaults to 'Linear'
        use_cache (bool, optional): Whether to use cached models. Defaults to False
        **kwargs: Additional keyword arguments
    """
 
    def train(self, X, y):
        pass

    def custom_predictor(self, X):
        return [1 for _ in range(len(X))]

class LinearRegressor(BaseRegressor):
    """
    Linear regression model.

    Args:
        data_splitter: Object containing train/test splits
        featurizer: Object that converts sequences to features
        model (str, optional): Name of model. Defaults to 'Linear'
        use_cache (bool, optional): Whether to use cached models. Defaults to False
        **kwargs: Additional keyword arguments
    """
    def __init__(self, data_splitter, featurizer, model='Linear', use_cache=False, **kwargs):
        super().__init__(data_splitter, featurizer, model, use_cache, **kwargs)
        
    def train(self, X, y):
        model = LinearRegression(
            fit_intercept=True,
            copy_X=True,
            n_jobs=10,
        )
        model.fit(X, y)
        self.model = model

    def custom_predictor(self, X):
        return self.model.predict(X)
    
class RandomForestRegressor(BaseRegressor):
    """
    Random Forest regression model.

    Args:
        data_splitter: Object containing train/test splits
        featurizer: Object that converts sequences to features
        model (str, optional): Name of model. Defaults to 'RandomForest'
        use_cache (bool, optional): Whether to use cached models. Defaults to False
        n_estimators (int, optional): Number of trees. Defaults to 100
        criterion (str, optional): Split criterion. Defaults to 'friedman_mse'
        max_depth (int, optional): Max tree depth. Defaults to None
        min_samples_split (int, optional): Min samples for split. Defaults to 2
        min_samples_leaf (int, optional): Min samples in leaf. Defaults to 1
        min_weight_fraction_leaf (float, optional): Min weight fraction in leaf. Defaults to 0.0
        max_features (float, optional): Max features to consider. Defaults to 1.0
        max_leaf_nodes (int, optional): Max leaf nodes. Defaults to None
        min_impurity_decrease (float, optional): Min impurity decrease. Defaults to 0.0
        bootstrap (bool, optional): Whether to bootstrap. Defaults to True
        oob_score (bool, optional): Whether to use out-of-bag score. Defaults to False
        n_jobs (int, optional): Number of parallel jobs. Defaults to 6
        random_state (int, optional): Random seed. Defaults to 1
        verbose (int, optional): Verbosity level. Defaults to 0
        warm_start (bool, optional): Whether to reuse solution. Defaults to False
        ccp_alpha (float, optional): Complexity parameter. Defaults to 0.0
        max_samples (int, optional): Max samples for bootstrap. Defaults to None
        **kwargs: Additional keyword arguments
    """
    def __init__(self, data_splitter, featurizer, model='RandomForest', use_cache=False,
                 n_estimators=100, 
                 criterion='friedman_mse',
                 max_depth=None,  
                 min_samples_split=2, 
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, 
                 max_features=1.0,
                 max_leaf_nodes=None, 
                 min_impurity_decrease=0.0, 
                 bootstrap=True, 
                 oob_score=False,
                 n_jobs=6, # change this based on number of cores 
                 random_state=1, 
                 verbose=0, 
                 warm_start=False, 
                 ccp_alpha=0.0,
                 max_samples=None,
                 **kwargs
                 ):
        
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

        super().__init__(data_splitter, featurizer, model=model, use_cache=use_cache, **kwargs)
    
    def train(self, X, y):
        self.model = RFRegressor(n_estimators=self.n_estimators,
                                 criterion=self.criterion,
                                 max_depth=self.max_depth,  
                                 min_samples_split=self.min_samples_split,
                                 min_samples_leaf=self.min_samples_leaf, 
                                 min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                 max_features=self.max_features, 
                                 max_leaf_nodes=self.max_leaf_nodes,
                                 min_impurity_decrease=self.min_impurity_decrease, 
                                 bootstrap=self.bootstrap, 
                                 oob_score=self.oob_score,
                                 n_jobs=self.n_jobs, 
                                 random_state=self.random_state, 
                                 verbose=self.verbose,
                                 warm_start=self.warm_start,
                                 ccp_alpha=self.ccp_alpha,
                                 max_samples=self.max_samples
                                )
        
        self.model.fit(X, y)

    def custom_predictor(self, X):
        return self.model.predict(X)

class RidgeRegressor(BaseRegressor):
    """
    Ridge regression model.

    Args:
        data_splitter: Object containing train/test splits
        featurizer: Object that converts sequences to features
        model (str, optional): Name of model. Defaults to 'Ridge'
        use_cache (bool, optional): Whether to use cached models. Defaults to False
        reg_coef (float, optional): Ridge regularization coefficient. If None, use CV. Defaults to None
        linear_model_cls (class, optional): Sklearn linear model class. Defaults to Ridge
        reg_coef_list (list, optional): List of regularization strengths for CV. Defaults to [0.1, 1.0, 2.0]
        **kwargs: Additional keyword arguments
    """
    # [TODO] edit cv to include modifiable splits
    def __init__(self, data_splitter, featurizer, model='Ridge', use_cache=False, reg_coef=None, linear_model_cls=Ridge, reg_coef_list=None, **kwargs):
        """
        Args:
            - reg_coef: Ridge regression coefficient. If none, then train with CV
            - linear_model_cls: sklearn linear model class
            - reg_coef_list: list of ridge regression regularization strength (default: [0.1, 1.0])
        """
        self.reg_coef = reg_coef
        self.linear_model_cls = linear_model_cls
        self.reg_coef_list = reg_coef_list if reg_coef_list is not None else [0.1, 1.0, 2.0]
        super().__init__(data_splitter, featurizer, model=model, use_cache=use_cache, **kwargs)
    
    def train(self, X, y):
        def spearman(y_pred, y_true):
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)

            y_pred = y_pred.reshape(-1)
            y_true = y_true.reshape(-1)
            
            if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
                return 0.0
            return ss.spearmanr(y_pred, y_true).correlation
        if self.reg_coef is None or self.reg_coef == 'CV':
            best_reg_coef, best_score = None, -np.inf
            for sample_reg_coef in self.reg_coef_list:
                model = self.linear_model_cls(alpha=sample_reg_coef)
                score = cross_val_score(model, X, y,
                                        cv=5,  
                                        scoring=make_scorer(spearman)).mean()
                if score > best_score:
                    best_reg_coef = sample_reg_coef
                    best_score = score
            self.model = self.linear_model_cls(alpha=best_reg_coef)
            self.model.fit(X, y)

    def custom_predictor(self, X):
        return self.model.predict(X)
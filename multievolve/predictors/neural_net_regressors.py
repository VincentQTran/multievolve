import torch, wandb
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import pandas as pd

from multievolve.utils.other_utils import performance_report, log_results
from multievolve.utils.data_utils import TorchDataProcessor

# Get the directory where the script is located
script_dir = os.path.dirname(__file__)

# Master Functions to Train and Evaluate Models
def run_nn_model_experiments(splits,
                             features,
                             models,  # Fcn, Cnn
                             experiment_name,
                             use_cache=False,
                             sweep_depth="standard",  # standard, custom, test
                             search_method="grid",  # grid, bayes, test
                             count=10,
                             show_plots=True
                             ):
    """Run neural network model experiments with hyperparameter sweeps.

    Args:
        splits (list): List of DataSplitter objects containing train/val/test splits
        features (list): List of feature types to use (e.g. ['onehot', 'esm'])
        models (list): List of model classes to run (e.g. [Fcn, Cnn]) 
        experiment_name (str): Name for the W&B experiment
        use_cache (bool, optional): Whether to cache results. Defaults to False.
        sweep_depth (str, optional): Sweep type - 'standard', 'custom', 'test'. Defaults to 'standard'.
        search_method (str, optional): Search method - 'grid', 'bayes', 'test'. Defaults to 'grid'.
        count (int, optional): Number of runs per sweep. Defaults to 10.
        show_plots (bool, optional): Whether to show matplotlib plots. Defaults to True.

    Returns:
        None: Results are logged to W&B

    Example:
        >>> splits = [DataSplitter(data, 'random')]
        >>> features = ['onehot']
        >>> models = [Fcn, Cnn]
        >>> run_nn_model_experiments(splits, 
        ...                         features, 
        ...                         models, 
        ...                         experiment_name='my_experiment',
        ...                         sweep_depth='selective',
        ...                         search_method='bayes',
        ...                         show_plots=True)
    """
        
    for split in splits:
        for feature in features:
            for model in models:
        
                """Define sweep configuration."""
                config_map = {
                    ("Fcn", "standard", "grid"): "fcn_standard_grid_sweep.yaml",
                    ("Fcn", "standard", "bayes"): "fcn_standard_bayes_sweep.yaml",
                    ("Fcn", "custom", "grid"): "fcn_custom_grid_sweep.yaml",
                    ("Fcn", "test", "test"): "fcn_test_sweep.yaml",
                    ("Cnn", "standard", "grid"): "cnn_standard_grid_sweep.yaml",
                    ("Cnn", "standard", "bayes"): "cnn_standard_bayes_sweep.yaml",
                    ("Cnn", "custom", "grid"): "cnn_custom_grid_sweep.yaml",
                    ("Cnn", "test", "test"): "cnn_test_sweep.yaml",
                }

                yaml_file = config_map.get((model.__name__, sweep_depth, search_method))
                if yaml_file is None:
                    print(
                        f"Invalid sweep configuration: model={model}, sweep_depth={sweep_depth}, search_method={search_method}."
                    )
                    return

                working_script_dir = script_dir

                # Assuming 'script_dir' is defined earlier in your code
                yaml_file_path = os.path.join(working_script_dir, "sweep_configs", yaml_file)

                with open(yaml_file_path, "r") as file:
                    sweep_config = yaml.safe_load(file)


                """initialize the sweep."""
                sweep_id = wandb.sweep(sweep=sweep_config, project=experiment_name)

                # Define a train function for hyperparameter sweeps with WANDB

                def train_function():
                    with wandb.init() as run:

                        # Grab config
                        config = run.config

                        # Specify model
                        instance = model(split, feature, use_cache=use_cache, config=config, show_plots=show_plots)

                        # Train and evaluate model
                        stat = instance.run_model()

                if search_method == "grid" or search_method == "test":
                    wandb.agent(sweep_id, train_function)
                elif search_method == "bayes":
                    wandb.agent(sweep_id, train_function, count=count)


# Neural network classes
class BaseNN(nn.Module):
    """Base neural network class implementing common functionality.
    
    This class provides the base implementation for neural network models including
    data loading, training loops, evaluation, and model saving/loading.

    Args:
        data_splitter: DataSplitter object containing train/val/test splits
        featurizer: Featurizer object for processing sequences
        nn_arch: Neural network architecture specification
        model (str): Model name identifier. Defaults to "Base"
        use_cache (bool): Whether to use model caching. Defaults to False
        show_plots (bool): Whether to show matplotlib plots. Defaults to True
        **kwargs: Additional keyword arguments

    Attributes:
        model_name (str): Name of the model
        featurizer: Featurizer object
        use_cache (bool): Whether caching is enabled
        kwargs (dict): Additional arguments
        nn_arch (str): Architecture specification string
        device (torch.device): Device to run model on (CPU/GPU)
        show_plots (bool): Whether to show matplotlib plots. Defaults to True
    Example:
        >>> splitter = DataSplitter(data, 'random')
        >>> featurizer = OneHotFeaturizer()
        >>> model = BaseNN(splitter, featurizer, [64,32], model='test', show_plots=True)
    """

    def __init__(self, data_splitter, featurizer, nn_arch, model="Base", use_cache=False, show_plots=True, **kwargs):
        super(BaseNN, self).__init__()

        # Set variables
        self.model_name = model
        self.featurizer = featurizer
        self.use_cache = use_cache
        self.kwargs = kwargs
        self.nn_arch = "-".join([str(x) for x in nn_arch])
        self.show_plots = show_plots

        # Setup data
        self.nn_data_processor = TorchDataProcessor(data_splitter, self.featurizer, self.kwargs["config"]["batch_size"])
        #[TODO] remove this and only process data once required

        self.split_method = self.nn_data_processor.split_name 

        # set model directory
        self.file_attrs = data_splitter.file_attrs
        self.file_attrs['model_dir'] = os.path.join(data_splitter.file_attrs["dataset_dir"], 'model_cache', data_splitter.file_attrs["dataset_name"])

        """Set variables."""
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("MPS available. Using Apple Silicon GPU for Neural Network.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("CUDA available. Using Nvidia GPU for Neural Network.")
        else:
            self.device = torch.device("cpu")
            print("Neither MPS nor CUDA is available. Using CPU for Neural Network.")

    def setup_model(self):
        """Set up the model by initializing hyperparameters and loading cached model if available."""

        # Retrieve hyperparameters from current run config
        self.set_hyperparams()

        # Define model
        self.file_attrs['model_name'] = (
                     self.split_method + " __ " + 
                     self.featurizer.name + " __ " + 
                     self.model_name + " __ " +
                     self.nn_arch + " __ " +
                     str(self.kwargs["config"]["learning_rate"]) + " __ " +
                     str(self.kwargs["config"]["batch_size"]) + " __ " +
                     self.kwargs["config"]["optimizer"]
                     )
        
        self.model_path = os.path.join(self.file_attrs['model_dir'], 'objects', f'{self.file_attrs["model_name"]}.pth')

        # Load model if available 
        if self.model_path is not None and os.path.exists(self.model_path) and self.use_cache:
            self.load_model(model_path=None)
            self.to(self.device)
        else: 
            self.to(self.device)

    def run_model(self, eval=True):
        """Run the full model training and evaluation pipeline.
        
        This method handles:
        1. Loading cached model if available
        2. Training the model if needed
        3. Evaluating on test set
        4. Saving model if caching enabled
        
        Returns:
            dict: Dictionary of model performance statistics
        """

        if self.model_path is not None and os.path.exists(self.model_path):
            model = self
            train_loss = self.train_loop_eval_mode(model)
            val_loss = self.val_loop(model)

        else:
            model = self
            
            # Train model

            for epoch in range(self.epochs):
                train_loss = self.train_loop(model)
                val_loss = self.val_loop(model)

                # Log data
                if wandb.run is not None:
                    wandb.log({"Train Loss": train_loss, "Val Loss": val_loss})

                # Check for early stopping
                if self.early_stopping_check(val_loss, epoch) == True:
                    break
                else:
                    continue

            # Save model
            if self.use_cache:
                self.save_model(model, model_path=None)

        # Test model
        if eval == True:
            return self.evaluate(model)
        else:
            return None
    
    def load_model(self, model_path=None):
        """Load a pre-trained model from disk.

        Args:
            model_path (str, optional): Path to model file. If None, uses default path.
        """

        # set location to load model
        model_path = self.model_path if model_path is None else model_path
        print(f"Loading model from {model_path}")
        # Load the trained model parameters
        self.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))

    def save_model(self, model, model_path=None):
        """Save model to disk.

        Args:
            model: Model to save
            model_path (str, optional): Path to save model to. If None, uses default path.
        """
        
        # set location to save model
        model_path = self.model_path if model_path is None else model_path

        dir_path = os.path.join(self.file_attrs['model_dir'], 'objects')
        # Check if the directory exists, create it if it doesn't
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Save the model
        print(f"Saving model to {self.model_path}")
        torch.save(model.state_dict(), self.model_path)

    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return x

    def train_loop(self, model):
        """Training loop for one epoch.
        
        Args:
            model: Model to train
            
        Returns:
            float: Average training loss for the epoch
        """
        model.train()
        total_train_loss = 0
        total_samples = 0

        # [TODO] new function to set up train loader if not already done
        if not hasattr(self, 'train_loader'):
            self.train_loader = self.nn_data_processor.setup_train_loader()

        for batch in self.train_loader:
            inputs, targets, __ = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = targets.unsqueeze(1)
            self.optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            total_train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            total_samples += inputs.size(0)

        train_loss = total_train_loss / total_samples

        return train_loss
    
    def train_loop_eval_mode(self, model):
        """Training loop in evaluation mode (no gradients).
        
        Args:
            model: Model to evaluate
            
        Returns:
            float: Average training loss
        """
        model.eval()
        with torch.no_grad():
            total_train_loss = 0
            total_samples = 0

            # [TODO] new function to set up train loader if not already done
            if not hasattr(self, 'train_loader'):
                self.train_loader = self.nn_data_processor.setup_train_loader()

            for batch in self.train_loader:
                inputs, targets, __ = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.unsqueeze(1)
                outputs = model(inputs)
                total_train_loss += self.criterion(outputs, targets).item()
                total_samples += inputs.size(0)

            train_loss = total_train_loss / total_samples

        return train_loss

    def val_loop(self, model):
        """Validation loop.
        
        Args:
            model: Model to evaluate
            
        Returns:
            float: Average validation loss
        """
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_samples = 0
            # [TODO] new function to set up val loader if not already done
            if not hasattr(self, 'val_loader'):
                self.val_loader = self.nn_data_processor.setup_val_loader()
                
            for batch in self.val_loader:
                inputs, targets, __ = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.unsqueeze(1)
                outputs = model(inputs)
                total_val_loss += self.criterion(outputs, targets).item()
                total_samples += inputs.size(0)

            val_loss = total_val_loss / total_samples

        return val_loss

    def evaluate(self, model):
        """Evaluate model on test set.
        
        Args:
            model: Model to evaluate
            
        Returns:
            dict: Dictionary of performance statistics
        """
        
        # Evaluate model, get metrics for validation and test set
        model.eval()

        stats_dict = {
            "val": {},
            "test": {}
        }
        
        loader_names = ["val", "test"]

        with torch.no_grad():

            # [TODO] new function to set up val and test loaders if not already done
            if not hasattr(self, 'val_loader'):
                self.val_loader = self.nn_data_processor.setup_val_loader()
            if not hasattr(self, 'test_loader'):
                self.test_loader = self.nn_data_processor.setup_test_loader()
                
            for index, loader in enumerate([self.val_loader, self.test_loader]):
                loader_name = loader_names[index]
                total_loss = 0
                total_samples = 0
                y = []
                y_pred = []
                original_sequences_list = []

                for batch in loader:
                    inputs, targets, original_sequences = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    targets = targets.unsqueeze(1)
                    outputs = model(inputs)
                    total_loss += self.criterion(outputs, targets).item()

                    # Move to CPU and convert to numpy
                    y.extend(targets.cpu().detach().numpy())
                    y_pred.extend(outputs.cpu().detach().numpy())
                    original_sequences_list.extend(original_sequences)

                    total_samples += inputs.size(0)


                # Reshape data and get correlation stats
                y = np.concatenate(y).ravel()
                y_pred = np.concatenate(y_pred).ravel()

                # Get stats
                stats_dict[loader_name] = performance_report(y, y_pred)

        # graph results for test set
        # Set the default parameters
        plt.rcParams['font.size'] = 7
        plt.rcParams['lines.linewidth'] = 0.5

        fig, ax = plt.subplots(figsize=(4, 3))
        
        # Mark data points that have activity less than 0 or greater than 1.2x the max experimental y value
        y_max = max(y.max(), y_pred.max()) * 1.2
        colors = np.where(y_pred > y_max, 'crimson', np.where(y_pred < 0, 'crimson', 'dodgerblue'))
        y_pred_adjusted = np.clip(y_pred, 0, y_max)

        # Scatter plot for main graph
        ax.scatter(y_pred_adjusted, y, c=colors, alpha=0.4, edgecolors='w', linewidth=0.5)

        # Draw x=y line
        ax.plot([0, y_max], [0, y_max], 'k--', linewidth=0.5)

        # Set labels and title for main graph
        ax.text(0.9, 0.1, f'Pearson r={stats_dict["test"]["Pearson r"]:.2f}', fontsize=7, ha='right', va='bottom', transform=ax.transAxes)
        ax.text(0.9, 0.2, f'Spearman r={stats_dict["test"]["Spearman r"]:.2f}', fontsize=7, ha='right', va='bottom', transform=ax.transAxes)
        ax.set_xlabel('Predicted Score', fontsize=7)
        ax.set_ylabel('True Score', fontsize=7)
        ax.set_title('Model Performance', fontsize=7)
        ax.set_xlim(0, y_max)

        # Display model parameters using legend
        model_params = self.file_attrs['model_name'].split('__')  # Assuming '|' separates different parameters
        param_text = '\n'.join(model_params)
        props = dict(boxstyle='square', facecolor='wheat', alpha=0.2)
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=7, verticalalignment='top', bbox=props)

        # Adjust tick parameters
        ax.tick_params(axis='both', which='major', labelsize=7)

        self.fig = fig

        if self.show_plots:
            plt.show()
        plt.close(fig)
            
        # Log data
        log_results(stats_dict, self)

        # Save predictions for test set as a table
        if self.use_cache:
            dir_path = os.path.join(self.file_attrs['model_dir'], 'results')
            # Check if the directory exists, create it if it doesn't
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            pred_results = pd.DataFrame({"original_sequences": original_sequences_list, "y": list(y), "y_pred": list(y_pred)})
            pred_results.to_csv(f"{dir_path}/{self.file_attrs['model_name']}.csv", index=False)

        return stats_dict['test']

    def early_stopping_check(self, val_loss, epoch):
        """Check if early stopping criteria are met.
        
        Args:
            val_loss (float): Current validation loss
            epoch (int): Current epoch number
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        # modify epoch count
        val_loss_delta = self.val_loss_min - val_loss
        if val_loss_delta > self.val_loss_delta_min:
            self.val_loss_min = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        # check epoch count
        if self.epochs_no_improve == self.patience:
            print(f"Early stopping after {epoch} epochs with {self.val_loss_min}.")
            return True
        else:
            return False

    def set_hyperparams(self):
        """Set model hyperparameters from config."""

        self.criterion = nn.MSELoss()
        self.lr = self.kwargs["config"]["learning_rate"]
        if self.kwargs["config"]["optimizer"] == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.kwargs["config"]["optimizer"] == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        self.epochs = self.kwargs["config"]["epochs"]

        # early stopping
        self.patience = 15
        self.val_loss_min = float("inf")
        self.val_loss_delta_min = 0.00001
        self.epochs_no_improve = 0  # initialize epochs_no_improve for early stopping

    def custom_predictor(self, X):
        """Make predictions on input data.
        
        Args:
            X: Input features
            
        Returns:
            numpy.ndarray: Model predictions
        """
        
        model = self
        inputs = torch.from_numpy(X.astype(np.float32)).to(self.device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            
        outputs_np = outputs.cpu().numpy()
        return outputs_np

    def predict(self, X, batch_size=10000):
        """Make predictions on sequences in batches.
        
        Args:
            X (list): List of sequences to predict
            
        Returns:
            numpy.ndarray: Array of predictions
        """
        batch_size = batch_size
        predictions = []
        
        # Process in batches
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            X_featurized = self.featurizer.featurize(batch)
            X_featurized = X_featurized.reshape(X_featurized.shape[0], -1)
            batch_predictions = self.custom_predictor(X_featurized)
            predictions.append(batch_predictions)

        return np.concatenate(predictions).ravel()

class Fcn(BaseNN):
    """Fully connected neural network model.
    
    Args:
        data_splitter: DataSplitter object containing train/val/test splits
        feature: Featurizer object for processing sequences
        model (str): Model name identifier. Defaults to "fcn"
        use_cache (bool): Whether to use model caching. Defaults to False
        show_plots (bool): Whether to show matplotlib plots. Defaults to True
        **kwargs: Additional keyword arguments including network architecture
        
    Example:
        >>> splitter = DataSplitter(data, 'random')
        >>> featurizer = OneHotFeaturizer()
        >>> model = Fcn(splitter, featurizer, config=config, use_cache=True, show_plots=True)
    """

    def __init__(self, data_splitter, feature, model="fcn", use_cache=False, show_plots=True, **kwargs):
        
        # Specify network architecture
        nn_arch = [kwargs["config"]["layer_size"]] * kwargs["config"]["num_layers"]

        super().__init__(data_splitter, feature, nn_arch, model, use_cache=use_cache, show_plots=show_plots, **kwargs)

        # [TODO] new function to extract input features if not already done

        X_train_feat_example = self.nn_data_processor.featurize([self.nn_data_processor.X_train[0]])[0]
        input_features = X_train_feat_example.flatten().shape[0]
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(input_features, nn_arch[0]))
        self.layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.layers.append(nn.Dropout(p=0.2))

        for i in range(0, len(nn_arch)):
            if i < len(nn_arch) - 1:
                self.layers.append(nn.Linear(nn_arch[i], nn_arch[i + 1]))
                self.layers.append(nn.LeakyReLU(negative_slope=0.2))
                self.layers.append(nn.Dropout(p=0.2))
            if i == len(nn_arch) - 1:
                self.layers.append(nn.Linear(nn_arch[i], 1))

        # set model hyperparameters
        self.setup_model()

    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
        return x

class Cnn(BaseNN):
    """Convolutional neural network model.
    
    Args:
        data_splitter: DataSplitter object containing train/val/test splits
        feature: Featurizer object for processing sequences
        model (str): Model name identifier. Defaults to "cnn"
        use_cache (bool): Whether to use model caching. Defaults to False
        show_plots (bool): Whether to show matplotlib plots. Defaults to True
        **kwargs: Additional keyword arguments including network architecture
        
    Example:
        >>> splitter = DataSplitter(data, 'random')
        >>> featurizer = OneHotFeaturizer()
        >>> model = Cnn(splitter, featurizer, use_cache=True, show_plots=True)
    """

    def __init__(self, data_splitter, feature, model="cnn", use_cache=False, show_plots=True, **kwargs):

        # Specify network architecture
        nn_arch = [kwargs["config"]["kernel_size"]] + [
            int(x) for x in kwargs["config"]["layersize_filtersize"].split("-")
        ]

        super().__init__(data_splitter, feature, nn_arch, model, use_cache=use_cache, show_plots=show_plots, **kwargs)

        # [TODO] new function to extract input features if not already done
        X_train_feat_example = self.nn_data_processor.featurize([self.nn_data_processor.X_train[0]])[0]
        protein_len = X_train_feat_example.shape[0]
        encoding_len = X_train_feat_example.shape[1]
        kernel_size_dim1, layers, out_channels = nn_arch

        in_channels = 1

        self.conv_layers = nn.ModuleList()

        for i in range(layers):
            conv2d_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size_dim1, encoding_len if i == 0 else 1),
                stride=(1, 1),
            )
            self.conv_layers.append(conv2d_layer)
            self.conv_layers.append(nn.LeakyReLU(negative_slope=0.2))
            in_channels = out_channels

        # Dynamically calculate the input size for the fully connected layer
        self._init_fc_layers(protein_len, encoding_len, out_channels)

        # set model hyperparameters
        self.setup_model()

    def _init_fc_layers(self, protein_len, encoding_len, out_channels):
        """Initialize fully connected layers.
        
        Args:
            protein_len (int): Length of protein sequence
            encoding_len (int): Length of sequence encoding
            out_channels (int): Number of output channels
        """
        # Dummy input for calculating size
        dummy_input = torch.randn(1, 1, protein_len, encoding_len)
        for layer in self.conv_layers:
            dummy_input = layer(dummy_input)

        output_size = dummy_input.view(dummy_input.size(0), -1).size(1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(output_size, 100)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = x.unsqueeze(1)  # add a channel dimension of 1 to the data
        for layer in self.conv_layers:
            x = layer(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

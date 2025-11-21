from multievolve.splitters import *
from multievolve.featurizers import *
from multievolve.predictors import *
from multievolve.proposers import *
from multievolve.utils import *

# dataset directory
main_dir = '../../data/benchmark/'
seq_dir = os.path.join(main_dir, 'sequences/')
datasets_dir = os.path.join(main_dir, 'datasets/')

summary = pd.read_csv(os.path.join(main_dir, 'dataset_summary.csv'))

for index, row in summary.iterrows():

    # default variables
    dataset_name, dataset_fname, sequence = receive_dataset_vars(row) # get dataset vars
    wt_file = retrieve_wt_file(dataset_name, seq_dir, sequence) # generate fasta file of sequence
    working_df_head, working_df_head_valid = preprocess_dataset(dataset_fname, datasets_dir, stringency='singles')

    # variables for training models
    protein_name = os.path.join(f"benchmark/", dataset_name)
    train_df = working_df_head_valid[['mutant','DMS_score','DMS_score_bin']].copy()
    
    # get feature
    feature = select_feature('onehot', protein_name)
    featurizers = [feature]     
    
    # get splitters 
    # generate split based on mutational load and do k-fold cross-validation
    splitters = []
    for max_train_mut_load in range(1,4,1):
        splitter = MutLoadProteinSplitter(protein_name, train_df, wt_file, use_cache=True, y_scaling=True, val_split=0.15)
        splitter.split_data(max_train_muts=max_train_mut_load, min_test_muts=4, k_folds=5)
        splitters = splitters + splitter.folds
    
    models = [Fcn]

    run_nn_model_experiments(splitters, 
                            featurizers, 
                            models, 
                            experiment_name=dataset_name,
                            use_cache=True,
                            sweep_depth='custom', 
                            search_method='grid',
                            count=1
                            )
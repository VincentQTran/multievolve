import ankh
import numpy as np
import torch

from multievolve.featurizers.base_featurizers import BaseFeaturizer

# alternate name: AnkhBaseFeaturizer
class AnkhFeaturizer(BaseFeaturizer):
    def __init__(self, 
                protein=None, 
                use_cache=False,
                model_version=None, 
                batch_size=968,
                model_type="ankh",
                **kwargs):
            
        super().__init__(model_type,protein, use_cache, **kwargs)

        self.batch_size = batch_size
        self.model_version = model_version

    def featurize_ankh(self, seqs):

        if self.model_version == 'large':
            self.model, self.tokenizer = ankh.load_large_model()
        elif self.model_version == 'base':
            self.model, self.tokenizer = ankh.load_base_model()
        else:
            raise ValueError(f"Invalid model version: {self.model_version}")
        self.model.eval()
        self.model.to(self.device)

        input_seqs = [list(seq) for seq in seqs]

        seq_batch = []

        for i in range(0, len(input_seqs), self.batch_size):
            batch = input_seqs[i:i + self.batch_size]
            outputs = self.tokenizer(
                batch, 
                add_special_tokens=True, 
                padding=True, 
                is_split_into_words=True, 
                return_tensors="pt",
            )
            outputs = {key: val.to(self.device) for key, val in outputs.items()}
            with torch.no_grad():
                embeddings = self.model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])
            seq_batch.append(embeddings['last_hidden_state'].mean(axis=1).cpu().numpy())

        return np.concatenate(seq_batch)

# alternate name: AnkhBaseEmbedFeaturizer
class AnkhBaseFeaturizer(AnkhFeaturizer):
    def __init__(self, 
                protein=None, 
                use_cache=False,
                model_version="base", 
                batch_size=968,
                model_type="ankh_base",
                **kwargs):
        super().__init__(protein, use_cache, model_version, batch_size, model_type, **kwargs)

    def custom_featurizer(self, seqs):

        X = self.featurize_ankh(seqs)
        return X

# alternate name: AnkhLargeEmbedFeaturizer
class AnkhLargeFeaturizer(AnkhFeaturizer):
    def __init__(self, 
                protein=None, 
                use_cache=False,
                model_version="large", 
                batch_size=968,
                model_type="ankh_large",
                **kwargs):
        super().__init__(protein, use_cache, model_version, batch_size, model_type, **kwargs)

    def custom_featurizer(self, seqs):
        X = self.featurize_ankh(seqs)
        return X
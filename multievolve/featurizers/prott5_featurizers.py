import ankh
import numpy as np
import torch

from multievolve.featurizers.base_featurizers import BaseFeaturizer
from transformers import T5Tokenizer, T5EncoderModel
import re

class ProtT5BaseFeaturizer(BaseFeaturizer):
    def __init__(self, 
                protein=None, 
                use_cache=False,
                model_version=None, 
                batch_size=968,
                model_type="ProtT5",
                **kwargs):
            
        super().__init__(model_type,protein, use_cache, **kwargs)

        self.batch_size = batch_size
        self.model_version = model_version

    def featurize_prott5(self, seqs):

        if self.model_version == 'prot_t5_xl_u50':
            self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(self.device)
        else:
            raise ValueError(f"Invalid model version: {self.model_version}")

        input_seqs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seqs]

        seq_batch = []

        for i in range(0, len(input_seqs), self.batch_size):
            batch = input_seqs[i:i + self.batch_size]
            # tokenize sequences and pad up to the longest sequence in the batch
            ids = self.tokenizer(batch, add_special_tokens=True, padding="longest")

            input_ids = torch.tensor(ids['input_ids']).to(self.device)
            attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

            # generate embeddings
            with torch.no_grad():
                embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)

            seq_batch.append(embeddings['last_hidden_state'].mean(axis=1).cpu().numpy())

        return np.concatenate(seq_batch)


class ProtT5_XL_U50_EmbedFeaturizer(ProtT5BaseFeaturizer):
    def __init__(self, 
                protein=None, 
                use_cache=False,
                model_version="prot_t5_xl_u50", 
                batch_size=968,
                model_type="ProtT5_XL_U50_Embed",
                **kwargs):
        super().__init__(protein, use_cache, model_version, batch_size, model_type, **kwargs)

    def custom_featurizer(self, seqs):

        X = self.featurize_prott5(seqs)
        return X
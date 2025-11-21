from jax_unirep.featurize import get_reps
from jax_unirep.utils import load_params

from multievolve.featurizers.base_featurizers import BaseFeaturizer

UNIREP_MODEL_SIZES = [1900, 256, 64]


class UnirepBaseFeaturizer(BaseFeaturizer):
    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=None,
        model_type="unirep",
        model_size=1900,
        **kwargs,
    ):
        super().__init__(model_type, protein, use_cache, **kwargs)
        self.model_locations = model_locations
        self.update_model_name(model_size)
        self.load_params()

    def update_model_name(self, model_size):
        # Validate model size
        assert model_size in UNIREP_MODEL_SIZES, "Model size must be 1900, 256, or 64."
        self.model_size = model_size
        self.model_type = self.model_type + str(model_size)

    def load_params(self):
        self.params = load_params(self.model_locations, self.model_size)[1]

    def custom_featurizer(self, seqs, **kwargs):
        h_avg, h_final, c_final = get_reps(
            seqs=seqs, params=self.params, mlstm_size=self.model_size
        )

        return h_avg

class EvotunedUnirepFeaturizer(UnirepBaseFeaturizer):
    def __init__(self, model_type="evotuned_unirep", **kwargs):
        super().__init__(model_type=model_type, **kwargs)
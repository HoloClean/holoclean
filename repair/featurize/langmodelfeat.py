import torch
from gensim.models import FastText

from dataset import AuxTables
from .featurizer import Featurizer


class LangModelFeaturizer(Featurizer):
    def specific_setup(self):
        self.name = 'LangModelFeaturizer'
        self.emb_size = 10
        self.all_attrs = self.ds.get_attributes()
        self.attrs_number = len(self.all_attrs)
        self.attr_language_model = {}
        raw_data = self.ds.get_raw_data()
        for attr in self.all_attrs:
            attr_corpus = list(zip(raw_data[attr].tolist()))
            model = FastText(attr_corpus, min_count=1, size=self.emb_size)
            self.attr_language_model[attr] = model
        # List[tuple(vid, attribute, '|||' separated domain)] sorted by vid.
        self.featurization_query_results = self._get_featurization_query_results()

    def _get_featurization_query_results(self):
        query = 'SELECT _vid_, attribute, domain FROM %s ORDER BY _vid_' % AuxTables.cell_domain.name
        return self.ds.engine.execute_query(query)

    def gen_feat_tensor(self, vid):
        assert(self.featurization_query_results[vid][0] == vid)
        input = self.featurization_query_results[vid]
        vid = int(input[0])
        attribute = input[1]
        domain = input[2].split('|||')
        attr_idx = self.ds.attr_to_idx[attribute]
        model = self.attr_language_model[attribute]
        tensor = torch.zeros(self.classes, self.attrs_number*self.emb_size)
        for idx, val in enumerate(domain):
            emb_val = model.wv[val]
            start = attr_idx*self.emb_size
            end = start+self.emb_size
            tensor[idx][start:end] = torch.tensor(emb_val)
        return tensor

    def feature_names(self):
        return ["{}_emb_{}".format(attr, emb_idx) for attr in self.all_attrs for emb_idx in range(self.emb_size)]

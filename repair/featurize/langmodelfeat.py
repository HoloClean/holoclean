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

    def gen_feat_tensor(self, input, classes):
        vid = int(input[0])
        attribute = input[1]
        domain = input[2].split('|||')
        attr_idx = self.ds.attr_to_idx[attribute]
        model = self.attr_language_model[attribute]
        tensor = torch.zeros(1, classes, self.attrs_number*self.emb_size)
        for idx, val in enumerate(domain):
            emb_val = model.wv[val]
            start = attr_idx*self.emb_size
            end = start+self.emb_size
            tensor[0][idx][start:end] = torch.tensor(emb_val)
        return tensor

    def create_tensor(self):
        query = 'SELECT _vid_, attribute, domain FROM %s ORDER BY _vid_' % AuxTables.cell_domain.name
        results = self.ds.engine.execute_query(query)
        tensors = [self.gen_feat_tensor(res, self.classes) for res in results]
        combined = torch.cat(tensors)
        return combined

    def feature_names(self):
        return ["{}_emb_{}".format(attr, emb_idx) for attr in self.all_attrs for emb_idx in range(self.emb_size)]

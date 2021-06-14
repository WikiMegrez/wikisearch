import torch
from transformers import BertTokenizer, BertModel
from wiki_search.dataset import Document


torch.set_grad_enabled(False)
MODEL_NAME = 'bert-base-cased'


class BertRanking(object):
    def __init__(self, device: str = 'cuda:0'):
        self.device = torch.device(device)
        self.model = BertModel.from_pretrained(MODEL_NAME).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    def _embed(self, text: str) -> torch.FloatTensor:
        tokens_pt = self.tokenizer(text, return_tensors='pt', max_length=512)
        tokens_pt = {k: v.to(self.device) for k, v in tokens_pt.items()}
        outputs = self.model(**tokens_pt)
        return outputs.pooler_output

    def score(self, query: str, doc: Document):
        query_z = self._embed(query)
        doc_z = self._embed(doc.raw_main_desc)
        score = (query_z * doc_z).sum()

        return score

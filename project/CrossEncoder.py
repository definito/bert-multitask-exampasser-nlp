from types import SimpleNamespace
import torch
from torch import nn
from datasets import SentenceClassificationDataset
from multitask_classifier import MultitaskBERT, process_batch
from optimizer import AdamW
from tokenizer import BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F

class CrossEncoder(nn.Module):

    def __init__(
            self,
            option='finetune', device='cuda', 
            hidden_dropout_prob=0.2,
            optimizer_params = {'lr': 2e-5},
            optimizer_class = torch.optim.AdamW,
                 ):
        
        super().__init__()
        # Init model
        config = {
            'hidden_dropout_prob': 0.001,
            'num_labels': {3: 0, 2: 1, 4: 2, 0: 3, 1: 4},
            'hidden_size': 768,
            'data_dir': '.',
            'option': 'pretrain',
            'local_files_only': False
            }


        config = SimpleNamespace(**config)

        self.device = device
        self.model = MultitaskBERT(config)
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.05, betas=[0.9, 0.999], eps=1e-06)


    def fit(self, train_dataloader: DataLoader):
            for sentence_pairs in tqdm(train_dataloader, desc="Iteration", smoothing=0.05):
                self.model.zero_grad()
                self.model.train()
                sst_ids, sst_mask, sst_labels = process_batch(sentence_pairs, self.device, "sst")
                sst_outputs = self.model.predict_sentiment(sst_ids, sst_mask)
                sst_loss = F.cross_entropy(sst_outputs, sst_labels.view(-1), reduction='mean')
                print(sst_loss)
                self.optimizer.zero_grad()
                sst_loss.backward()
                self.optimizer.step()
            
    

if __name__ == '__main__':
    sentence_pairs = [
        ["The sun is shining. [SEP] It's a beautiful day.", 0.85, 1],
        ["Cats are independent animals. [SEP] Dogs are loyal companions.", 0.65, 2],
        ["Learning a new language can be challenging. [SEP] Practice is key to language acquisition.", 0.75, 3],
        ["Books are a source of knowledge. [SEP] Reading enhances intellectual growth.", 0.90, 4]
    ]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)
    sentence_pairs = SentenceClassificationDataset(sentence_pairs, None, tokenizer)
    sentence_pairs_dataloader = DataLoader(sentence_pairs, shuffle=True, batch_size=4,
                                        collate_fn=sentence_pairs.collate_fn, num_workers=4)
    cross_encoder = CrossEncoder(device = 'cpu')
    cross_encoder.fit(train_dataloader=sentence_pairs_dataloader)

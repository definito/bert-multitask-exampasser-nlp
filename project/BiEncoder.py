from types import SimpleNamespace
import torch
from torch import nn
from datasets import SentencePairDataset
from multitask_classifier import MultitaskBERT, process_batch
from optimizer import AdamW
from tokenizer import BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F

class BiEncoder(nn.Module):

    def __init__(
            self,
            train_dataloader: DataLoader,
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
        
        self.model = MultitaskBERT(config)
        self.model = self.model.to(device)
        self.device = device

        self.optimizer = AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.05, betas=[0.9, 0.999], eps=1e-06)

    def fit(self, train_dataloader: DataLoader):
        for sentence_pairs in tqdm(train_dataloader, desc="Iteration", smoothing=0.05):
            self.model.zero_grad()
            self.model.train()
            training_steps = 0
            sts_id1, sts_mask1, sts_id2, sts_mask2, sts_labels = process_batch(sentence_pairs, self.device, "sts")
            embeddings_sentence_1 = self.model.forward(sts_id1, sts_mask1)
            embeddings_sentence_2 = self.model.forward(sts_id2, sts_mask2)
            similarity = F.cosine_similarity(embeddings_sentence_1, embeddings_sentence_2)
            loss = nn.MSELoss()
            output = loss(similarity, sts_labels.float())
            self.optimizer.zero_grad()
            output.backward()
            self.optimizer.step()
            # Pass the embeddings through the sentiment classifier to get predictions
            
    

if __name__ == '__main__':
    sentence_pairs = [
        ["The sun is shining.", "It's a beautiful day.", 0.85, 1],
        ["Cats are independent animals.", "Dogs are loyal companions.", 0.65, 2],
        ["Learning a new language can be challenging.", "Practice is key to language acquisition.", 0.75, 3],
        ["Books are a source of knowledge.", "Reading enhances intellectual growth.", 0.90, 4]
    ]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)
    sentence_pairs = SentencePairDataset(sentence_pairs, None, tokenizer)
    sentence_pairs_dataloader = DataLoader(sentence_pairs, shuffle=True, batch_size=4,
                                        collate_fn=sentence_pairs.collate_fn, num_workers=4)
    bi_encoder = BiEncoder(device = 'cpu', train_dataloader=sentence_pairs_dataloader)
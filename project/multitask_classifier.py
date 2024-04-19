from io import open

# !!!NEED TO DELETE!!!! JUST FOR DEV!!!!
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm


from tokenizer import BertTokenizer

from datasets import SentenceClassificationDataset, SentencePairDataset,\
    load_multitask_data

from evaluation import test_model_multitask, model_eval_multitask

from smart_pytorch import SMARTLoss, SMARTLossSPair
from loss import kl_loss, sym_kl_loss, js_loss

TQDM_DISABLE=True


# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

EARLY_STOP_COUNT = 5

def check_device(*tensors):
    for tensor in tensors:
        if tensor is not None and not tensor.is_cuda:
            print(f"Warning: Tensor {tensor} is not on GPU, it's on {tensor.device}")



import random

def maybe_swap(embeddings_1, embeddings_2):
    if random.random() > 0.5:
        return embeddings_1, embeddings_2
    else:
        return embeddings_2, embeddings_1
    
import time

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

class ComplexBlock(nn.Module):
    """
    A custom block implementing several architectural patterns:
    1. Layer Normalization similar to He's method.
    2. Residual connections as in ResNet.
    3. Multi-Layer Perceptron (MLP) with optional dropout and customizable activation.
    
    The block is structured with three main layers, each comprising Layer Normalization, 
    followed by an MLP, dropout, and activation.
    """
    
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        """
        Initializes the ComplexBlock.

        Parameters:
        - in_dim (int): Input dimension.
        - hidden_dim (int): Hidden layer dimension.
        - out_dim (int): Output dimension.
        - activation (callable, optional): Activation function. Default is ReLU.
        - dropout (float, optional): Dropout rate. Default is 0.1.
        """
        super(ComplexBlock, self).__init__()
        
        # Residual connection
        self.residual = nn.Linear(in_dim, out_dim)
        
        # Define each layer with LayerNorm, MLP, Dropout, and Activation
        self.layer1 = self._make_layer(in_dim, hidden_dim)
        self.layer2 = self._make_layer(hidden_dim, hidden_dim)
        self.layer3 = self._make_layer(hidden_dim, out_dim)
        
        # Activation & Dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)


    def _make_layer(self, in_dim, out_dim):
        """
        Utility function to create a layer with LayerNorm, MLP, and Dropout.
        """
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.Linear(in_dim, out_dim),
        )
    
    def forward(self, x):
        """
        Forward pass through the block.
        """
        residual = self.residual(x)
        
        # Pass input through each layer and add the residual connection at the end
        x = self.layer1(x)
        x = self.dropout(self.activation(x))
        
        x = self.layer2(x)
        x = self.dropout(self.activation(x))
        
        x = self.layer3(x)
        x = self.dropout(x)
        
        x += residual
        return x
    
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=1)
        else:
            sum_embeddings = (x * mask.unsqueeze(-1)).sum(dim=1)
            mean_pooled = sum_embeddings / mask.sum(dim=1, keepdim=True)
            return mean_pooled

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        
        # Initialize BERT
        # self.bert = BertModel.from_pretrained('/home/bzkurs51/Downloads/bert-base-uncased', local_files_only=config.local_files_only)
        self.bert = BertModel.from_pretrained('bert-base-uncased', local_files_only=config.local_files_only)
        
        # Set BERT parameters to trainable or not based on the configuration
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        
        # Sentiment classification layers
        self.sentiment_classifier = nn.Sequential(
            ComplexBlock(self.bert.config.hidden_size, 1024, 256),
            nn.Linear(256, N_SENTIMENT_CLASSES)  # 5 sentiment classes
        )

        # Paraphrase detection layers
        self.paraphrase_classifier = nn.Sequential(
            ComplexBlock(self.bert.config.hidden_size * 4, self.bert.config.hidden_size * 4, 256),  # Multiplied by 4 for concatenation, difference, and element-wise multiplication
            nn.Linear(256, 1)
        )
        
        # Semantic Textual Similarity layers
        self.similarity_classifier = nn.Sequential(
            ComplexBlock(self.bert.config.hidden_size * 4, self.bert.config.hidden_size * 4, 256),  # Multiplied by 4 for concatenation, difference, and element-wise multiplication
            nn.Linear(256, 1)
        )
        
        self.logits = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0], requires_grad=True))
        self.mean_pooling_layer = MeanPooling()
        
        # MLM head
        self.mlm_head = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
                
        self.label_words = ['terrible', 'bad', 'neutral', 'good', 'excellent']
        self.label_word_ids = torch.tensor([tokenizer.convert_tokens_to_ids(word) for word in self.label_words])
        self.smart_weight = 0.05
        
    def forward(self, input_ids=None, attention_mask=None, task = 'classification', if_emb = False, bert_outputs=None):
        """
        Encodes a batch of sentences into embeddings using BERT model.
        
        The function retrieves the embedding of the [CLS] token after passing 
        through a random dropout and a dense layer.
        
        Args:
        - input_ids (torch.Tensor): The tokenized input sentences.
        - attention_mask (torch.Tensor): The attention masks for input sentences.
        
        Returns:
        - hidden_states (torch.Tensor): The encoded embeddings for input sentences.
        """
        if if_emb == False:
            bert_outputs = self.bert(input_ids, attention_mask)
        hidden_states = bert_outputs['last_hidden_state']
        
        if task == 'classification':
            mp_hidden_states = self.mean_pooling_layer(hidden_states, mask=attention_mask)
            return mp_hidden_states
        elif task == 'mlm':
            return self.mlm_head(hidden_states)
        else:
            raise ValueError("Invalid task")
        

    def get_weights(self):
        """
        Retrieves the learnable weights for multi-task training.
        """
        return F.softmax(self.logits, dim=0)
    
    # sentence MASK SST
    def predict_sentiment(self, input_ids=None, attention_mask=None, emb = None):
        
        # forward pass
        if emb is None:
            mlm_logits = self.forward(input_ids, attention_mask)
        else:
            outputs = self.bert(attention_mask = attention_mask, embedding_output = emb)
            mlm_logits = self.forward(if_emb=True, bert_outputs=outputs)
            
        mask_logits = self.mlm_head(mlm_logits) #[batch_size, vocab_size]
        
        # The logits corresponding to the five label words are further extracted
        selected_logits = mask_logits[:, self.label_word_ids]
        
        return selected_logits
    
    def predict_sentiment_smartloss(self, input_ids, attention_mask, labels):
        emb = self.bert.embed(input_ids)
        def eval(emb):
            return self.predict_sentiment(attention_mask=attention_mask, emb = emb)
        smart_loss_fn = SMARTLoss(eval_fn = eval, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
        state = eval(emb)
        loss = F.cross_entropy(state, labels)
        loss = loss + self.smart_weight * smart_loss_fn(emb, state)
        return loss
    
    def predict_paraphrase(self,
                    input_ids_1= None, attention_mask_1= None,
                    input_ids_2= None, attention_mask_2= None, emb1 = None, emb2 = None):
        """
        Predicts whether two batches of sentences are paraphrases of each other using the trained paraphrase classifier.

        Returns:
        - torch.Tensor: The predicted paraphrase logits for each pair of sentences.
        """
        if emb1 == None:
            # Encode the sentences into embeddings
            embeddings_1 = self.forward(input_ids_1, attention_mask_1)
            embeddings_2 = self.forward(input_ids_2, attention_mask_2)
        else:
            # pass embed+encoded
            emb1 = self.bert(attention_mask = attention_mask_1, embedding_output = emb1)
            emb2 = self.bert(attention_mask = attention_mask_2, embedding_output = emb2)
            # pass my model
            embeddings_1 = self.forward(if_emb=True, bert_outputs=emb1)
            embeddings_2 = self.forward(if_emb=True, bert_outputs=emb2)
        
        embeddings_1, embeddings_2 = maybe_swap(embeddings_1, embeddings_2)

        # Combine embeddings using various strategies for richer representations
        # - Concatenate the embeddings
        # - Take the absolute difference of embeddings
        # - Perform element-wise multiplication of embeddings
        concat_embeddings = torch.cat((embeddings_1, embeddings_2, 
                                torch.abs(embeddings_1 - embeddings_2), 
                                embeddings_1 * embeddings_2), dim=1)
        # Predict whether the sentences are paraphrases
        # paraphrase_logits = self.para_only(self.para_sts_shared(concat_embeddings))
        paraphrase_logits = self.paraphrase_classifier(concat_embeddings)
        
        return paraphrase_logits
    
    def predict_paraphrase_smartloss(self,
                        input_ids_1, attention_mask_1,
                        input_ids_2, attention_mask_2,labels):
        
        embeddings_1 = self.bert.embed(input_ids_1)
        embeddings_2 = self.bert.embed(input_ids_2)
        
        if attention_mask_1 is None or attention_mask_2 is None:
            raise ValueError("Attention masks should not be None")

        def eval_(embeddings_1, embeddings_2):
            return self.predict_paraphrase(None, attention_mask_1, None, attention_mask_2, embeddings_1, embeddings_2)
        smart_loss_fn = SMARTLossSPair(eval_fn = eval_, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
        state = eval_(embeddings_1, embeddings_2)
    
        loss = F.binary_cross_entropy_with_logits(state.view(-1), labels.float(), reduction='mean')
        
        loss = loss + self.smart_weight * smart_loss_fn(embeddings_1,embeddings_2, state)
        return loss


    def predict_similarity(self,
                        input_ids_1, attention_mask_1,
                        input_ids_2, attention_mask_2, emb = None):
        """
        Predicts the semantic textual similarity (STS) between two batches of sentences.

        Returns:
        - torch.Tensor: The similarity scores for each pair of sentences, scaled to be in the range [0, 5].
        """
        
        # Encode the sentences into embeddings and apply activation
        emb1 = self.forward(input_ids_1, attention_mask_1)
        emb2 = self.forward(input_ids_2, attention_mask_2)
        
        emb1, emb2 = maybe_swap(emb1, emb2)
        
        concat_embeddings = torch.cat((emb1, emb2, 
                                    torch.abs(emb1 - emb2), 
                                    emb1 * emb2), dim=1)
        
        return torch.sigmoid(self.similarity_classifier(concat_embeddings)) * N_SENTIMENT_CLASSES

    

def save_model(model, optim, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"Saved the model to {filepath}")
    
def process_batch(batch, device, task_name):
        '''Helper function to unpack and move batch data to the desired device.'''
        if task_name == "sst":
            ids, mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels']
            # sents, sent_ids = batch['sents'], batch['sent_ids']
        else:
            ids, mask, labels = batch['token_ids_1'], batch['attention_mask_1'], batch['labels']
            ids2, mask2 = batch['token_ids_2'], batch['attention_mask_2']
            return ids.to(device), mask.to(device), ids2.to(device), mask2.to(device), labels.to(device)
        return ids.to(device), mask.to(device), labels.to(device) #sents, sent_ids


def train_multitask(args, tokenizer):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    # Load Datasets
    with Timer() as t:# Use Timer to record the time cost
        # Load the datasets for Multi-Task Training
        sst_train_data, num_labels,para_train_data, sts_train_data = \
            load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')
    print(f"Loaded all datasets. Time taken: {t.interval:.2f} seconds")
    
    with Timer() as t:
        sst_train_data = SentenceClassificationDataset(sst_train_data, args, tokenizer)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args, tokenizer)
        para_train_data = SentencePairDataset(para_train_data, args, tokenizer)
        para_dev_data = SentencePairDataset(para_dev_data, args, tokenizer)
        sts_train_data = SentencePairDataset(sts_train_data, args, tokenizer)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, tokenizer, isRegression=True)
    print(f"Loaded multi-task data sentence. Time taken: {t.interval:.2f} seconds")
    
    # Dataloader
    with Timer() as t:
        sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sst_train_data.collate_fn, num_workers=4)
        
        
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn, num_workers=4)
        para_train_dataloader = DataLoader(para_train_data, shuffle=True, \
            batch_size=args.batch_size, collate_fn=para_train_data.collate_fn, num_workers=4)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, \
            batch_size=args.batch_size, collate_fn=para_dev_data.collate_fn, num_workers=4)
        sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, \
            batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn, num_workers=4)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, \
            batch_size=args.batch_size, collate_fn=sts_dev_data.collate_fn, num_workers=4)
    
        
    
    print(f"Loaded all dataloaders. Time taken: {t.interval:.2f} seconds")
    
    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option,
              'local_files_only': args.local_files_only}

    config = SimpleNamespace(**config)
    
    model = MultitaskBERT(config)
    model = model.to(device)
    model.label_word_ids = model.label_word_ids.to(device)
    
    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=[0.9, 0.999], eps=1e-06)

    print("------Supervised Multi-task Training with SST, Quora, STS------")

    best_dev_acc = 0
    not_improving_count = 0
    EVAL_INTERVAL = 1  # epochs
    from itertools import zip_longest
    
    with Timer() as t:
    # Check if pretrained multitask model exists
        if False:
        # if os.path.exists(args.filepath):
            model.load_state_dict(torch.load(args.filepath)["model"])
            print(f"Loaded pretrained multi-task model from {args.filepath}")
            paraphrase_accuracy_dev, para_y_pred_dev, para_sent_ids_dev,\
            sentiment_accuracy_dev,sst_y_pred_dev, sst_sent_ids_dev,\
            sts_corr_dev, sts_y_pred_dev, sts_sent_ids_dev = model_eval_multitask(sst_dev_dataloader,para_dev_dataloader, sts_dev_dataloader, model, device)
            dev_acc = (paraphrase_accuracy_dev + sentiment_accuracy_dev + sts_corr_dev)/3
            print(f"Multi-task dev acc :: {dev_acc :.3f}")
        else:
            # Multi-task Training loop
            print("Starting multi-task training...")
            for epoch in range(args.epochs):
                print(f"Training Epoch number {epoch} begins")
                model.train()
                step_count = 0
                total_loss = 0
                num_batches = 0
                
                dataloader_iterator = zip_longest(sst_train_dataloader, para_train_dataloader, sts_train_dataloader)
                # dataloader_iterator = zip(sst_train_dataloader, para_train_dataloader, sts_train_dataloader)
                                
                for (sst_batch, para_batch, sts_batch) in tqdm(dataloader_iterator, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                    optimizer.zero_grad()
                    weights = model.get_weights()
                    active_losses = 0
                    train_loss = 0
                    if sst_batch is not None:
                        # Process SST task, class label 0,1,2,3,4
                        sst_ids, sst_mask, sst_labels,*_ = process_batch(sst_batch, device, "sst")
                        sst_loss = model.predict_sentiment_smartloss(sst_ids, sst_mask, sst_labels)
                        train_loss += weights[0] * sst_loss
                        active_losses += 1
                    if para_batch is not None: 
                        # Process Paraphrase Task(1/0)
                        para_id1, para_mask1, para_id2, para_mask2, para_labels = process_batch(para_batch, device, "para")
                        para_loss = model.predict_paraphrase_smartloss(para_id1, para_mask1, para_id2, para_mask2, para_labels)
                        train_loss += weights[1] * para_loss
                        active_losses += 1
                    if sts_batch is not None:
                        # Process STS Task, (0.0-5.0)
                        sts_id1, sts_mask1, sts_id2, sts_mask2, sts_labels = process_batch(sts_batch, device, "sts")
                        sts_outputs = model.predict_similarity(sts_id1, sts_mask1, sts_id2, sts_mask2)
                        sts_loss = F.mse_loss(sts_outputs.squeeze(), sts_labels.view(-1).float(), reduction='mean') / args.batch_size
                        train_loss += weights[2] * sts_loss
                        active_losses += 1
                    if active_losses > 0:  # Avoid dividing by zero
                        train_loss /= active_losses  # avg the loss
                    train_loss.backward()
                    optimizer.step()
                    
                    total_loss += train_loss.item()
                    num_batches += 1
                    step_count += 1
                    # Evaluate every EVAL_INTERVAL steps 改成epochs
                if epoch % EVAL_INTERVAL == 0:
                    print("train acc => ")
                    paraphrase_accuracy, para_y_pred, para_sent_ids,\
                    sentiment_accuracy,sst_y_pred, sst_sent_ids,\
                    sts_corr, sts_y_pred, sts_sent_ids= model_eval_multitask(sst_train_dataloader,para_train_dataloader, sts_train_dataloader, model, device)
                    print("dev acc => ")
                    paraphrase_accuracy_dev, para_y_pred_dev, para_sent_ids_dev,\
                    sentiment_accuracy_dev,sst_y_pred_dev, sst_sent_ids_dev,\
                    sts_corr_dev, sts_y_pred_dev, sts_sent_ids_dev = model_eval_multitask(sst_dev_dataloader,para_dev_dataloader, sts_dev_dataloader, model, device)
                    
                    train_acc = (paraphrase_accuracy + sentiment_accuracy + sts_corr) / 3
                    dev_acc = (paraphrase_accuracy_dev + sentiment_accuracy_dev + sts_corr_dev)/3

                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                        save_model(model, optimizer, args, config, args.filepath)
                        print(f"Higher (avg) accuracy achieved at step {epoch}: {best_dev_acc:.3f}, saved model to {args.filepath}")
                        not_improving_count = 0
                    else:
                        not_improving_count += 1
                        
                    if not_improving_count >= EARLY_STOP_COUNT:
                        print(f"Accuracy didn't improve for {not_improving_count} consecutive evaluations, early stopping!")
                        break
                    
            print(f"Epoch {epoch}: train loss :: {total_loss:.3f}, train acc :: {train_acc:.3f}, dev acc :: {dev_acc:.3f}") 
    print(f"Time taken: {t.interval:.2f} seconds")

def test_model(args, tokenizer):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device, tokenizer)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64 can fit a 12GB GPU', type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=1e-3)
    parser.add_argument("--local_files_only", action='store_true')
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    with Timer() as t:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=args.local_files_only)
    print(f"Loaded the tokenizer in advance. Time taken: {t.interval:.2f} seconds")
    train_multitask(args, tokenizer)
    test_model(args, tokenizer)

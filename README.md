# Information

Repository standard: https://github.com/paperswithcode/releasing-research-code

Qumeng's Note: Please be aware that due to my current network limitations, I can only access the internet through a mobile hotspot with very poor signal. As a result, I am unable to upload our two best models, A and B, at this time. Model A is our best-performing model on SST, and Model B is our best-performing model on average; we have ultimately submitted Model B to the TA. However, I have saved both models locally and they will be available for any future requests.

2023.9.12 Note: We are now updated the best models. Please check the "Pre-trained models" sections.

# `exampasser` nlp project

This repository is a course project for the Natural Language Processing(DNLP) course in the SS23 semester at the University of Göttingen and is maintained by four team members.

We combined some existing techniques, fine-tuned a BERT-base model, and achieved an average accuracy of **0.749** on three tasks, which is improve **49.5%** from the baseline.

You can use this repository to check our results, or to further train the BERT base model.

Members: Yasir Yasir, *Qumeng Sun, Debwashis Borman.

>📋  Optional: include a graphic explaining your approach/main result, BibTeX entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model in the paper, run this command:

```train
python -u multitask_classifier.py
```

There are all arguments you can use:

### Model Training Input Parameters

#### Data Input Files

- `--sst_train`: Path to the training data for the SST task. The default is `data/ids-sst-train.csv`.
- `--sst_dev`: Path to the development data for the SST task. The default is `data/ids-sst-dev.csv`.
- `--sst_test`: Path to the test data for the SST task. The default is `data/ids-sst-test-student.csv`.
- `--para_train`: Path to the training data for the paraphrase task. The default is `data/quora-train.csv`.
- `--para_dev`: Path to the development data for the paraphrase task. The default is `data/quora-dev.csv`.
- `--para_test`: Path to the test data for the paraphrase task. The default is `data/quora-test-student.csv`.
- `--sts_train`: Path to the training data for the STS task. The default is `data/sts-train.csv`.
- `--sts_dev`: Path to the development data for the STS task. The default is `data/sts-dev.csv`.
- `--sts_test`: Path to the test data for the STS task. The default is `data/sts-test-student.csv`.

#### Output Files

- `--sst_dev_out`: Path to save the SST development output. The default is `predictions/sst-dev-output.csv`.
- `--sst_test_out`: Path to save the SST test output. The default is `predictions/sst-test-output.csv`.
- `--para_dev_out`: Path to save the paraphrase development output. The default is `predictions/para-dev-output.csv`.
- `--para_test_out`: Path to save the paraphrase test output. The default is `predictions/para-test-output.csv`.
- `--sts_dev_out`: Path to save the STS development output. The default is `predictions/sts-dev-output.csv`.
- `--sts_test_out`: Path to save the STS test output. The default is `predictions/sts-test-output.csv`.

#### General Parameters

- `--seed`: Random seed for reproducibility. The default is `11711`.
- `--epochs`: Number of training epochs. The default is `10`.
- `--option`: Specifies the training option. Choices are `pretrain` (BERT parameters are frozen) or `finetune` (BERT parameters are updated). The default is `pretrain`.
- `--use_gpu`: Flag to enable GPU usage. If set, the model will be trained on GPU.
- `--local_files_only`: Flag to use only local files and avoid downloading.

#### Hyperparameters

- `--batch_size`: Batch size for training. The default is `64`. 
- `--hidden_dropout_prob`: Dropout rate for the hidden layer. The default is `1e-3`.
- `--lr`: Learning rate. The default is `1e-5`. For `pretrain`, the suggested learning rate is `1e-3`, and for `finetune`, it is `1e-5`.

If you want to get the average performance model:

```
python -u multitask_classifier.py --option finetune --epochs 12 --batch_size 16 --hidden_dropout_prob 0.2 --lr 1e-05 --use_gpu
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

In our `evaluation.py`, you can find the eval function provide by the course. Also, when you run the `multitask_classifier.py`, you also automatically runs the evaluation function.

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pre-trained models here: https://1drv.ms/f/s!AoH0J35av8mcl98YA2VxqMWsn029Lg?e=y5pgtX

- The best model A&B was trained on SST, Quora, and STS datasets, you can find the parameters we used in the `Experiments` section.
- You can find them in the folder called "Best Models", `finetune-5-3e-05-multitask.pt` is the model A, the other one is the model B.

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance:

| Fine-tune                          | SST   | Quora | STS   | Avg.  |
| ---------------------------------- | ----- | ----- | ----- | ----- |
| BERT-base                          | 0.509 | 0.682 | 0.311 | 0.501 |
| BERT-base w/ multi-task            | -     | -     | -     | -     |
| BERT-base w/ unsup-SimCSE          | 0.524 | 0.696 | 0.238 | 0.486 |
| BERT-base w/ S-BERT                | -     | -     | -     | -     |
| BERT-base w/ multi-task, prompt (Model A)    | 0.548 | 0.834 | 0.863 | 0.748 |
| Final Model w/ full Quora data (Model B) | 0.524 | 0.880 | 0.841 | 0.749 |

Due to limited time and experience, we were not able to test all the models individually, so there are some missing in the chart, you can refer to our experiment section for more confidence. There, you can see the accuracies achieved for each feature, and their comparisons.

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.

## Contributing

We welcome contributions to the `exampasser` project! Whether it's bug reports, feature suggestions, or direct contributions to the code, we appreciate all forms of collaboration. Here's how you can contribute:

#### Reporting Issues

1. Check if the issue has already been reported.
2. If not, create a new issue and provide a detailed description of the problem.

#### Code Contributions

1. Fork the repository.
2. Clone your fork locally: `git clone https://gitlab.com/.../exampasser.git`
3. Create a new branch for your feature or bug fix: `git checkout -b feature/your-feature-name` or `git checkout -b fix/your-fix-name`
4. Make your changes.
5. Commit your changes: `git commit -m "Your commit message"`
6. Push to your fork: `git push origin feature/your-feature-name` or `git push origin fix/your-fix-name`
7. Create a pull request against the `main` branch of the original `exampasser` repository.

#### Code Style

Please follow the coding style. This ensures consistency and readability.

#### Testing

Before submitting a pull request, make sure to run all tests to ensure that your changes do not break existing functionality.

All contributors will be acknowledged in this README and in any academic publications that use or refer to this code.

#### Documentation

If you're adding a new feature, make sure to also update the README or any relevant documentation.

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

>📋  Pick a license and describe how to contribute to your code repository.

## Methodology

Our method is based on four main strategies: 

1. Multiple-Task Learning with Sentence-BERT, responsible by Qumeng;
2. Trans-Encoder, responsible by Yasir;
3. Prompt Learning, responsible by Qumeng;
4. PALs, responsible by DB.

Our group members will introduce the strategies they implemented individually. If you want to see the citations(used resources) of our project, you should check the `Contributions` section.

※Note: Due to performance issue, we did not use the Trans-Encoder in our final model. Due to the time issue, we did not use the PALs in our final model.

#### The Model

We use the pre-trained BERT-base model. We added three layers to deal with three different tasks. Each layer contains a `ComplexBlock`, which includes `Residual connection`, `Dropout`, and `LayerNorm`. We also have a `mlm_head` function to deal with the prompt things in the SST task. In the `forward`, we use the mean pooling of  `'last_hidden_state'` of BERT. 

For the multiple-task training, we also have a parameter to represent the three tasks' learnable weights of three losses. You can call the `get_weights` function to use it.

We changed the function `predict_sentiment` for sentence-level prompt learning to fit it. It only contains a forward pass and a `mlm_head` process. In the other two tasks, we use `sentence-BERT(SBERT)`.

We changed the function `predict_paraphrase` and `predict_similarity` to fit our SMART method.

#### Multiple-Task Learning

We trained on the three datasets provided (SST, PARA, STS). We used the learnable parameter `logits` to adjust the weights for the three tasks. 
$$
L_{\text{Total}} = W[0] \cdot L_{\text{SST}} + W[1] \cdot L_{\text{PARA}} ＋ W[2] \cdot L_{\text{STS}}
$$
Since the amount of data in the provided data is very different for the three datasets (see below), we used zip_longest to ensure all the data is fully utilized.

We observe a massive difference between our train loss and dev loss in the SST task. Typically when we get 85%+ training accuracy we only get 52% accuracy in the test set. In addition, the PARA task has a larger amount of data and is likewise more prone to overfitting. Therefore, we then applied the SST and PARA tasks on a regularization method called `SMART`.

```
Loaded 8544 train examples from data/ids-sst-train.csv
Loaded 141498 train examples from data/quora-train.csv
Loaded 6040 train examples from data/sts-train.csv
Loaded 1101 train examples from data/ids-sst-dev.csv
Loaded 20212 train examples from data/quora-dev.csv
Loaded 863 train examples from data/sts-dev.csv
```

$$
\text{min}_\theta\mathcal F(\theta)=\mathcal L(\theta)+\lambda_S\mathcal R_S(\theta)
$$



Simply put, SMART is a regularization method that lets a model learn the noise that benefits it. It adds noise at the sentence embedding level, then compares the difference between the different outputs, and uses KL divergence to measure the difference between them. Learning the appropriate noise minimizes the KL divergence between them. Since the existing SMART code we used only supports a single sentence input, we extended it so that it can add different disturbances to each of the two sentences and obtain a uniform loss. The way to apply it is to add a regularization term after the original loss function. 

The code about this part is stored in `smart_pytorch.py` and `loss.py`, and we use `predict_sentiment_smartloss` and `predict_ paraphrase_smartloss`. Inside them, we call the `SMARTLoss` class to implement the SMART method.

In addition, we randomly scramble the order of the input sentences in the PARA and STS tasks in order to obtain more order-robust results. We also implemented early stopping during this training process, and we set the early listening hyperparameter to be 5 epochs.

#### Trans-Encoder
We used a single pre-trained BERT model to create four models. Two of these models acted as cross-encoders, and two acted as bi-encoders. The cross-encoders and bi-encoders were used to generate pseudo-labels, which were then used to train the other models. This process was iterated for three cycles. Each encoder had a different loss function.

The approach did not work well in our case. One possible explanation for the failure is that the bi-encoders and cross-encoders were created from the same base model, so they did not have any difference in knowledge. This means that there was no knowledge to be transferred, and only the last layers of the two models were different.

If we had more time, one approach we could have tried is to train a RoBERTa model and distill its knowledge into the BERT model. RoBERTa is a newer language model that has been shown to be more effective than BERT on a variety of tasks. By distilling the knowledge from [RoBERTa](https://arxiv.org/abs/1907.11692) into BERT, we could have potentially improved the performance of the BERT model.

#### Prompt Learning

In our final model, we applied prompt learning in the SST task. However, this is not a classical MLM-based learning but rather a prediction at the sentence level to accommodate another of our features, namely SBERT. Specifically, we expanded the original SST data into the following style:
$$
(S_{input};L_{true}) \rightarrow (S_{input} , \text{it is [MASK].}S_{example_1}, \text{it is terrible.}...)
$$
`it is` in the formula is a prompt sentence, terrible is a prompt label words, and `S_example` is an example sentence. We tested a prompt sentence based on the likelihood of watching the movie again (I will ... watch it again), but it didn't work as well as the standard prompt sentence we use now. So, we still keep the simpler standard prompt sentence. Our label words use:
$$
\text{[terrible, bad, neutral, good, excellent]}
$$
Since SST is a classification task with five categories, we randomly set five example sentences so the model can better understand it. These five example sentences are the first five we randomly selected from the SST training set. This operation did not significantly improve the performance of SST but improved the performance of the other two tasks. Please refer to the experimental section for details.

The unique feature of our method is to make predictions at the sentence level instead of the token level. We first add the prompt sentence and five examples to the sentences one by one. After converting this long sentence through BERT's tokenizer, use the Mean Pooling result of `[CLS]` of this long sentence to predict the mask word. We modified several functions in `datasets.py` and `multitask_classifier.py` to achieve this operation.

#### PALs*

Implemented by Debwashis, we didn't end up using this feature since we didn't have enough time to merge this into our main branch. But it brought us 1~2% performance improvement in our previous experiments. 
Performance on previous Model: 
```
dev sentiment acc :: 0.511
dev paraphrase acc :: 0.764
dev sts corr :: 0.799
```

## Experiments

In this section, we will demonstrate the impact of each strategy by showing the performance with and without them. Also, we will record the processing of finding the best hyperparameters.

#### Best Model(s)

Our model currently has some uncertainty on the performance. You may get different results even if you have the same codes. Also, since this is a course project, we do not have unlimited testing time. It is quite possible that you found better models in your experiments. 

In our **best model A**, we got 0.748 average accuracy(`finetune-5-3e-05-multitask_best.pt`). 

```
dev sentiment acc: 0.548
dev paraphrase acc: 0.834
dev sts corr: 0.863
```

The hyperparameters are five epochs, 16 batch size, 0.2 dropout prob, 3e-05 lr. We are getting different results around 0.747 with the same model and hyperparameters.

In our **best model B**, we got an average accuracy of 0.749, which is higher than the best model A.

```
dev sentiment acc: 0.524
dev paraphrase acc: 0.880
dev sts corr: 0.0.841
```

We found a better model with a lower learning rate of `7e-06`. We trained a model with an average accuracy of 0.747, and later, we further trained this model with a lower learning rate. The hyperparameters of the first train are ten epochs, 16 batch size, 1e-05 learning rate, and 0.2 dropout probability. The second train: 10 epochs, 32 batch size, 7e-06 learning rate, 0.2 dropout rate.

The other hyperparameters are always the same, we do not have enough time to test them.

EARLY_STOP_COUNT = 5

BERT_HIDDEN_SIZE = 768

self.smart_weight = 0.05

##### PCG

| Model                 | SST   | Quora | STS   |
| --------------------- | ----- | ----- | ----- |
| Pre-train Model       | 0.315 | 0.624 | 0.177 |
| Pre-train Model w PCG | 0.316 | 0.624 | 0.177 |
| Fine-tune Model       | 0.510 | 0.680 | 0.323 |
| Fine-tune Model w PCG | 0.509 | 0.682 | 0.311 |

##### Sentence-BERT

| Model                                    | SST   | Quora | STS   |
| ---------------------------------------- | ----- | ----- | ----- |
| Fine-tune BERT-base w/ linear classifier | 0.520 | 0.718 | 0.355 |
| Fine-tune Model w/ cosine similarity     | 0.499 | 0.776 | 0.704 |
| Fine-tune Model w/ SBERT                 | 0.489 | 0.774 | 0.793 |
| Fine-tune Model w/ SBERT + Mean Pooling  | 0.502 | 0.770 | 0.801 |

Please note that SBERT here includes our modification. We share the same method on PARA and STS tasks. Also, we found that using an extra dense layer in the forward function did not help our performance. So, we dropped the extra Complex Block, dropout, and dense layer in our forward function in the last week of our project.

##### Complex Block

C-Block is a simple way to add extra blocks. It contains MLP, Dropout, Residual connections, and Layer Norm. It brought a great improvement to Quora and STS tasks.

| Model                 | SST   | Quora | STS   |
| --------------------- | ----- | ----- | ----- |
| Fine-tune Model       | 0.507 | 0.718 | 0.661 |
| Fine-tune Model w/ CB | 0.505 | 0.764 | 0.718 |

##### SimCSE

###### Unsupervised

| Model                          | SST   | Quora | STS   |
| ------------------------------ | ----- | ----- | ----- |
| Pre-train Model                | 0.336 | 0.625 | 0.410 |
| Pre-train Model w Unsup-SimCSE | 0.354 | 0.620 | 0.803 |
| Fine-tune Model                | 0.498 | 0.718 | 0.648 |
| Fine-tune Model w Unsup-SimCSE | 0.520 | 0.720 | 0.780 |

I found that SimCSE conflicts with Sentence-BERT. When we use them together, the results will not increase anymore. But if we use them separately, they both benefit our model performance. Because SimCSE's time cost is worse, we decided not to use SimCSE anymore. We only use S-BERT in our final model.

###### Supervised

| Model                        | SST   | Quora | STS   |
| ---------------------------- | ----- | ----- | ----- |
| Fine-tune Model              | 0.523 | 0.790 | 0.797 |
| Fine-tune Model w Sup-SimCSE | 0.490 | 0.801 | 0.793 |

Supervised SimCSE did not bring any noticed improvement to us, which is quite superisely.

###### Supervised + Teacher-Student

| Model                              | SST   | Quora | STS   |
| ---------------------------------- | ----- | ----- | ----- |
| Fine-tune Model                    | 0.500 | 0.798 | 0.777 |
| Fine-tune Model w Sup-SimCSE + T-S | 0.505 | 0.708 | 0.799 |

Teacher-Student setup is inspired by the Distillation Model. We train a model with SimCSE separately, and combine the loss with this teacher model when we do the multi-task training.

##### Fake Random Dropout

| Model                   | SST   | Quora | STS   |
| ----------------------- | ----- | ----- | ----- |
| Pre-train Model         | 0.336 | 0.625 | 0.410 |
| Pre-train Model w FRD   | 0.354 | 0.620 | 0.803 |
| Fine-tune Model         | 0.498 | 0.718 | 0.648 |
| Fine-tune Model w/o FRD | 0.520 | 0.720 | 0.780 |

Fake Random Dropout(FRD) is a way to use dropout layers with different dropout probabilities in the forward function of our model. We found it helpful working with the SimCSE. It can improve the accuracy of SST tasks while harming the other two tasks. After a lot of testing, we decided to drop it from our final model together with SimCSE.

##### Additional Datasets

We tested FP and MS datasets with supervised SimCSE, and they did bring some improvement to the STS and PARA(Quora) task, but we deleted them for simplicity.

```
Only Multi-task
dev sentiment acc :: 0.504
dev paraphrase acc :: 0.764
dev sts corr :: 0.709
+FP
dev sentiment acc :: 0.507
dev paraphrase acc :: 0.742 (↓)
dev sts corr :: 0.658 (↓)
+MS
dev sentiment acc :: 0.521 (↑)
dev paraphrase acc :: 0.784 (↑)
dev sts corr :: 0.658 (↓)
```

We tried to use the IMDB datasets to do the further pre-training, because the SST task is a amount of movie reviews, which is in the same domain of IMDB datasets. But, we did not get the expected result, it does not help us.

| Model                                    | SST   | Quora | STS   |
| ---------------------------------------- | ----- | ----- | ----- |
| Fine-tune Model                          | 0.528 | 0.824 | 0.855 |
| Fine-tune Model w IMDB further pre-train | 0.520 | 0.811 | 0.849 |

##### Trainable Dirichlet Distribution Prior

We tried to add a Dirichlet prior on SST data, but the results was not good, so we did not use it in our final model.

| Model                              | SST   | Quora | STS   |
| ---------------------------------- | ----- | ----- | ----- |
| Fine-tune Model                    | 0.516 | 0.806 | 0.848 |
| Fine-tune Model w/ Dirichlet Prior | 0.446 | 0.812 | 0.855 |

##### Prompt Learning

After our tests, Prompt Learning at the token level does not help us improve performance. So we ended up choosing a sentence-level promt that doesn't add the `[SEP]` tag. We tested another combination once, which was "I will... watch this movie again". Considering the frequency of vocabulary in BERT, our word selection is ['never', 'not', 'maybe', 'probably', 'definitely']. We got good results, but we still chose the default prompt.

| Model                                                  | SST   | Quora | STS   |
| ------------------------------------------------------ | ----- | ----- | ----- |
| Fine-tune Model                                        | 0.521 | 0.834 | 0.848 |
| Fine-tune Model w token-level prompt                   | 0.500 | 0.823 | 0.849 |
| Fine-tune Model w sentence-level + [SEP]               | 0.520 | 0.828 | 0.844 |
| Fine-tune Model w sentence-level(likelihood of review) | 0.520 | 0.832 | 0.860 |
| Fine-tune Model w sentence-level prompt                | 0.529 | 0.831 | 0.856 |

##### SMART

This approach has allowed us to improve our stability, and our model can now arrive at an accuracy of 0.747 very consistently, and even 0.749 under multiple tests. Before this, our model struggled to exceed 0.741 accuracy most of the time.

## Contributions description

In this section, our group members will detail their contributions to this project and highlight the critical parts with the emoji 💡.

Note: The user who commits many codes, called "Mike-7777777," is also Qumeng. His GitHub account was not switched when committing to VSC, leading to a situation like this.

#### Muhammad Hammad

After speaking with Muhammad, we learned that, for some reason, he could not continue with the project, so he left our group in May and that the work he had completed was reading a paper.

#### Qumeng Sun

I will list all the contributions related to me and highlight the important ones.

Note: I frequently use GPT to write and clean the codes. If I do not mention the existing resources used, then it means that my implementation is based only on the original paper of the method I mentioned.

###### Find, Read, and Implement Papers

- Read and implement the `How to Fine-Tune BERT for Text Classification?`, Sun et al. LNAI 2020 https://gitlab.gwdg.de/exampasser/nlp/-/work_items/12
- Read and implement the `MTRec`, Bi et al. ACL 2022 https://gitlab.gwdg.de/exampasser/nlp/-/work_items/17
- 💡Find, read, and implement the Supervised + Unsupervised `SimCSE`, Gao et al. EMNLP 2021 https://gitlab.gwdg.de/exampasser/nlp/-/issues/30
- Find, read, and implement the `SRWGD`, Kumar et al. 2023 https://gitlab.gwdg.de/exampasser/nlp/-/issues/28
- 💡Find, read, and implement the `Sentence-BERT`, Reimers et al. EMNLP 2019 https://gitlab.gwdg.de/exampasser/nlp/-/issues/35 
- 💡Find, read, and implement the `Prompt Learning on BERT`, ACL | IJCNLP 2021, https://gitlab.gwdg.de/exampasser/nlp/-/issues/38
- 💡Find, and read, and implement the `SMART`, ACL 2020, Microsoft AI. I did not have time to implement it eventually.
- Find, read, and implement the additional datasets. In issue https://gitlab.gwdg.de/exampasser/nlp/-/issues/29 and https://gitlab.gwdg.de/exampasser/nlp/-/issues/34

###### Coding

- Implementation of task one, in https://gitlab.gwdg.de/exampasser/nlp/-/commit/ce84634acbb53617f7d81dc5e975e5fbfc1f3c8e, https://gitlab.gwdg.de/exampasser/nlp/-/commit/742973519252c38b137eae9238a1b2727c4de041, and https://gitlab.gwdg.de/exampasser/nlp/-/commit/70fe29458b262bcd0d92a8fe370203de6fec6740.
- Fix several bugs in the official GitHub codes
  - The dataset selection bug, https://gitlab.gwdg.de/exampasser/nlp/-/commit/8aa1ada7321d509e331661b40642e86f8b7f7825
  - The data loading bug, https://gitlab.gwdg.de/exampasser/nlp/-/commit/5366096a6104240f70ffb99a6dfe1b982dabd164.
  - GWDG usage problem(cannot download the pretrain model automatically on the cloud).
  - Disk Quota Exceeded Error https://gitlab.gwdg.de/exampasser/nlp/-/issues/33
- Improvement of the official code.
  - Load the BERT tokenizer only once rather than multiple times, decreasing the time cost. https://github.com/truas/minbert-default-final-project/pull/2
- Implementation of the three heads, which are sentiment classifier, paraphrase classifier, and similarity classifier. https://gitlab.gwdg.de/exampasser/nlp/-/commit/5366096a6104240f70ffb99a6dfe1b982dabd164
- 💡Implementation of `(Weighted)Multiple-Task Learning`. https://gitlab.gwdg.de/exampasser/nlp/-/issues/22
  - Included learnable weights.
  - Rewrite the batch process in `process_batch` function, increase the code reuse.
  - Use `SMART` on the SST task, adopting codes from [smart-pytorch](https://github.com/archinetai/smart-pytorch).
- Implementation of `ComplexBlock` https://gitlab.gwdg.de/exampasser/nlp/-/commit/37f340cc9b24b8092a8c54284c32464e6a66473f
  - It contains
    - `Residual connection`, initially raised by Kaiming He in the [paper](https://arxiv.org/abs/1512.03385) 
    - `Dropout`, initially raised by Srivastava, in [paper](https://dl.acm.org/doi/10.5555/2627435.2670313) 
    - `LayerNorm`, initially raised by this [paper](https://arxiv.org/abs/1607.06450), was used in front of MLP is presented by Kaiming He in the [paper](https://arxiv.org/abs/1603.05027)
- Implementation of Multiple Datasets loading(datasets.py). https://gitlab.gwdg.de/exampasser/nlp/-/issues/29
  - NLI for `SimCSE` https://gitlab.gwdg.de/exampasser/nlp/-/commit/501d145c0cb9b72491c2508559c1f71fd409fdaf
  - Wiki1M for `SimCSE` https://gitlab.gwdg.de/exampasser/nlp/-/commit/37f340cc9b24b8092a8c54284c32464e6a66473f
- Implementation of `Timer`, counting the time cost of each part of our model training.
- Implementation of `MeanPooling` class.
- 💡Implementation of `Prompt Learning`, in https://gitlab.gwdg.de/exampasser/nlp/-/commit/16dbca0d7e14f04a7c17e544810b6681712ab24b.
  - I use a sentence-level prompt, which is novel (to my knowledge). So, in this method, I modify the `dataset.py` to add example sentences to every sentence sample and make the model predict the `[MASK]` word in this sentence. But, in the forward function, I use mean pooling of `[CLS]`, which means we are considering this task at the sentence level, not the token or word level, which brings us the best improvement. I tried another prompt method that does it at the token level with a classic MLM setup, but the performance did not increase.
  - We already use the `Sentence-BERT`, so it may benefit us when processing everything on the sentence level.
    I use the first five sentences for the example, so it can be further improved if we have time to do more tests or even implement the automatic prompt search in the original paper.


###### Others

- Open and maintain the GitLab repository
  - Write the Wiki of the project and meeting records.
  - Write the README file.  (Except for what is explicitly stated to be written for someone else.)
---
### Yasir

#### Papers 
- Read and Implement Simple Contrastive Learning of Sentence Embeddings. Gao et al. [2021]
- Read and Implement 5.3 Gradient surgery for multi-task learning. Yu et al. [2020].
- Read the BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning. Stickland and Murray [2019] Task actions 
- Read and implement Trans-Encoder: Unsupervised sentence-pair modelling through self- and mutual-distillations. Credits to Qumeng for finding this paper. 
  

#### Coding
- 💡Trans-Encoder: Unsupervised sentence-pair modelling through self- and mutual-distillations
  - The paper is implementation of pseudo labelling model, where 1 model generates the labels for the the next model and the roles are switched in the next phase. The BiEncoder generates the embeddings for two sentences seprately which are used along with cosine similarity function to calculate the pesudo labels. Whereas the Cross Encoder takes in sentences pairs separeted with **[SEP]** token and the results are passed into an MLP.
  - I used a subset of STS dataset 2013.tsv only. The parsing of data and generation of dataloader [done here](https://gitlab.gwdg.de/exampasser/nlp/-/commit/b97d4dcc5f5fb72686cde8c81758f9ed821c6690) 
  - Created implementation of BI-Encoder and Cross Encoder model from scratch. Using the [cross encoder](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/cross_encoder/CrossEncoder.py) source code and from sentence transformer library and BI encoder using the [sentence transformer code](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py).
  - The official implementation of the [paper](https://arxiv.org/abs/2109.13059) [by Amazon](https://github.com/amzn/trans-encoder) has been of great help. 
  - Add [process_batch](https://gitlab.gwdg.de/exampasser/nlp/-/commit/f820c4227ec5c737108a388b4069d07283ded6c8) for unlabeled datasets.
- 💡Gradient Surgery for Multi-Task Learning.
  - The paper proposes projecting conflicting gradients" (PCGrad) to mitigate the problem of conflicting gradients. PCGrad works by projecting the gradient of each task onto the normal plane of the gradient of any other task that has a conflicting gradient.
  - Worked on implementing the gradient surgery for paper but found an [implementation here](https://github.com/WeiChengTseng/Pytorch-PCGrad).
- General Contributions
  - Tried out difference loss functions such as Cosine embeddings loss. Cosine similarity loss. Kullbeck liebler divergence for optimizing the embeddings. in Bi Encoder and Cross encoder models. 
  - Identified a few bugs  such as:
    - The accuracy calculation of semantic similarity.
    - The model was not training for all the data points we have.The for loop was running only for the smallested dataset we have. Qumeng applied the fix [here](https://gitlab.gwdg.de/exampasser/nlp/-/commit/a5faea71630e2a50adb4805fade0bbc04d377144)



### Debwashis 

#### Papers
- Efficient Natural Language Response Suggestion for Smart Reply Henderson et al. [2017]
- BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning. Stickland and Murray [2019] Task actions

#### Implementation:
-  PAL's: Though the paper contains a completely different implementation approach- https://github.com/AsaCooperStickland/Bert-n-Pals, the main idea has been taken and implemented from the scratch for our model. It doesn't modify our main model architechture, however a improved performance has achieved(1%-2%). However, as other implementations were occuring frequently and due to timeframe for submission merging process couldn't be done to avoid potential errors. 
 
- Summary: 
  - This work explores how to adapt a single large base model to work with multiple tasks. They use the BERT model (Bidirectional Encoder Representations from Transformers, Devlin et al., 2018) as their base pre-trained model. Pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, including the GLUE benchmark. 

    -  ‘Projected Attention Layer’ (PAL), a low-dimensional multi-head attention layer that is added in parallel to normal BERT layers.
    -  They introduce a novel method for scheduling training, where we sample tasks proportional to their training set size at first, and de-emphasize training set size as training proceeds. 
    - They perform an empirical comparison of alternative adaptation modules for self-attention-based architectures.
- Results from the paper: 
  
  - Result: 
    - 1st table:
    - BERT-base: This is the base BERT model, with 8x the number of parameters compared to the rest. Its average GLUE score was 79.6.
    - Shared: A model where all parameters are shared except the final projection to output space, with a GLUE score of 79.9.
    - Top Projected Attention (Top Proj. Attn.): This model adds parameters at the 'top' of the BERT model, with a GLUE score of 79.6.
    - Projected Attention Layers (PALS) with 204: This model adds task-specific function in parallel with each BERT layer and achieved a GLUE score of 80.4.

    - In 2nd table, the performance of these models using different multi-task training methods (Proportional Sampling, Square Root Sampling, and Annealed Sampling) is shown. Again, the Projected Attention Layers (PALS) method, in particular PALS with 204 hidden layers and Annealed Sampling, showed superior performance with a GLUE score of 81.7. 

- Other:
  - The paper: Efficient Natural Language Response Suggestion for Smart Reply Henderson et al. [2017]- this is not relevant with our project goal. So after reading it, we decided to drop it from our implementation task.  

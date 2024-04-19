# Log

For reference(the SOTA scores):
- SST (Stanford Sentiment Treebank) 90.37%
- Quora(Question Answering on Quora Question Pairs) 92.3%
- SemEval STS Benchmark Dataset 92.9%

```

0812 Local test 1

Due to the [Errno 122] Disk quota exceeded problem on cloud, we only test 2 epochs on our local GPU.

python -u multitask_classifier.py --option * --epochs * --batch_size 16 --hidden_dropout_prob 0.1 --lr 2e-05 --use_gpu

---

Local main branch+PCG *pretrain* 2 epochs
dev sentiment acc :: 0.316
dev paraphrase acc :: 0.624
dev sts corr :: 0.177

Local main branch+PCG *finetune* 2 epochs
dev sentiment acc :: 0.509
dev paraphrase acc :: 0.682
dev sts corr :: 0.311

---

Local main branch-PCG *pretrain* 2 epochs
dev sentiment acc :: 0.315
dev paraphrase acc :: 0.624
dev sts corr :: 0.177

Local main branch-PCG *finetune* 2 epochs
dev sentiment acc :: 0.510
dev paraphrase acc :: 0.680
dev sts corr :: 0.323

---

```

```

0812 Local test 2

Local main brach+PCG *pretrain* 2 epochs + pc_adam.step()
dev sentiment acc :: 0.316
dev paraphrase acc :: 0.624
dev sts corr :: 0.177

Local main brach+PCG *finetune* 2 epochs + pc_adam.step()
dev sentiment acc :: 0.509
dev paraphrase acc :: 0.682
dev sts corr :: 0.311

Local main brach+PCG *finetune* 2 epochs + pc_adam.step() + _project_conflicting (Improve computational efficiency)
dev sentiment acc :: 0.516
dev paraphrase acc :: 0.666
dev sts corr :: 0.323
13:12 - 14:40

+ g_j_norm_squared
dev sentiment acc :: 0.510
dev paraphrase acc :: 0.647
dev sts corr :: 0.314
14:45 - 15:50

```

```

0812 Local test 3

Local main brach-PCG *pretrain* 2 epochs -sts
dev sentiment acc :: 0.317
dev paraphrase acc :: 0.624
dev sts corr :: -0.009
16:51-16:55

Local main brach-PCG *finetune* 2 epochs -sts
dev sentiment acc :: 0.500
dev paraphrase acc :: 0.688
dev sts corr :: -0.057

Local main brach-PCG *finetune* 2 epochs -sts +simcse_sts_unsupervised
dev sentiment acc :: 0.497
dev paraphrase acc :: 0.679
dev sts corr :: 0.032
17:16 - 17:22

Local main brach-PCG *finetune* 2 epochs + simcse_sts_unsupervised (括号内是应该对比的原成绩,对照组)
dev sentiment acc :: 0.524 (0.510)
dev paraphrase acc :: 0.696 (0.680)
dev sts corr :: 0.238(0.323)

-PCP +last_hidden_state[:, 0, :]
dev sentiment acc :: 0.530
dev paraphrase acc :: 0.722
dev sts corr :: 0.367

-PCP +last_hidden_state[:, 0, :] + simple simcse_unsup_loss
dev sentiment acc :: 0.511
dev paraphrase acc :: 0.704
dev sts corr :: 0.358

-PCP +last_hidden_state[:, 0, :] + simple simcse_unsup_loss + _1_2
dev sentiment acc :: 0.506
dev paraphrase acc :: 0.696
dev sts corr :: 0.326

-PCP +last_hidden_state[:, 0, :] + simple simcse_unsup_loss 融合两种
dev sentiment acc :: 0.509
dev paraphrase acc :: 0.671
dev sts corr :: 0.253

-PCP +last_hidden_state[:, 0, :] + simcse_unsup_loss 中文 + _1 _1 + concat
dev sentiment acc :: 0.520
dev paraphrase acc :: 0.677
dev sts corr :: 0.234

-PCP +last_hidden_state[:, 0, :] + simcse_unsup_loss 中文 + _1 _2 + concat 
dev sentiment acc :: 0.509
dev paraphrase acc :: 0.669
dev sts corr :: 0.244
second time test (8-13)
dev sentiment acc :: 0.494
dev paraphrase acc :: 0.701
dev sts corr :: 0.261

... dense act
dev sentiment acc :: 0.514
dev paraphrase acc :: 0.723
dev sts corr :: 0.350
新对照组(8-13 test no simcse)
dev sentiment acc :: 0.510
dev paraphrase acc :: 0.725
dev sts corr :: 0.357

...dense act + dropout 0.2
dev sentiment acc :: 0.514
dev paraphrase acc :: 0.723
dev sts corr :: 0.350

...dense act + dropout layer 0.2(加了一个在multitaskBert)
dev sentiment acc :: 0.509
dev paraphrase acc :: 0.718
dev sts corr :: 0.340

去掉dolayer
dev sentiment acc :: 0.518
dev paraphrase acc :: 0.718
dev sts corr :: 0.346

在内部(loss方法处)加dolayer
dev sentiment acc :: 0.520
dev paraphrase acc :: 0.721
dev sts corr :: 0.341

up + trainbale weights of loss function
dev sentiment acc :: 0.507
dev paraphrase acc :: 0.717
dev sts corr :: 0.326

+ 先wiki1m再multitask
--option finetune --epochs 2 --batch_size 32 --hidden_dropout_prob 0.2 --lr 2e-05 
dev sentiment acc :: 0.443
dev paraphrase acc :: 0.703
dev sts corr :: 0.238

+ 使用了loss method而不是重写 使用了subset而不是全集
dev sentiment acc :: 0.517
dev paraphrase acc :: 0.724
dev sts corr :: 0.303

+ 同上 + 把zero放前面
dev sentiment acc :: 0.496
dev paraphrase acc :: 0.727
dev sts corr :: 0.303

+ 同上 + 添加了2个不同的dropout层
dev sentiment acc :: 0.520
dev paraphrase acc :: 0.718
dev sts corr :: 0.355
test 2(test_2.pt)
dev sentiment acc :: 0.509
dev paraphrase acc :: 0.736
dev sts corr :: 0.343

+ comment out multi-task phrase(test_3.pt) 

+ use new similarity comupute formula test_4
70
dev sentiment acc :: 0.520
dev paraphrase acc :: 0.720
dev sts corr :: 0.780

+ finetune 20 epochs
71
dev sentiment acc :: 0.520
dev paraphrase acc :: 0.720
dev sts corr :: 0.780

+ pretain 2 epochs
72
dev sentiment acc :: 0.354
dev paraphrase acc :: 0.620
dev sts corr :: 0.803

无unsup的对比组：
finetune 2 epochs - unsupervised simcse
dev sentiment acc :: 0.498
dev paraphrase acc :: 0.718
dev sts corr :: 0.648
pretrain
dev sentiment acc :: 0.336
dev paraphrase acc :: 0.625
dev sts corr :: 0.410

有unsup+改了sst函数
pt
dev sentiment acc :: 0.379
dev paraphrase acc :: 0.625
dev sts corr :: 0.266
ft

+ 增加了 复杂块 改了整体函数sst para sts finetune
dev sentiment acc :: 0.502
dev paraphrase acc :: 0.748
dev sts corr :: 0.787

+ PCG finetune
dev sentiment acc :: 0.499
dev paraphrase acc :: 0.746
dev sts corr :: 0.782

+ PCG pretrain
dev sentiment acc :: 0.433
dev paraphrase acc :: 0.749
dev sts corr :: 0.787

+ fake random dropout in forward + PCG
dev sentiment acc :: 0.507
dev paraphrase acc :: 0.718
dev sts corr :: 0.661

- cb layer in forward (keep fake random dropout in forward, and PCG)
dev sentiment acc :: 0.505
dev paraphrase acc :: 0.764
dev sts corr :: 0.718

so, - cb in forward is better

同上 + cb layer - dense layer

同上 + cb layer - dense layer 把fake random dropout -> count random dropout
dev sentiment acc :: 0.493
dev paraphrase acc :: 0.684
dev sts corr :: 0.666

把drop比率改回来（最开始是+0和+0.1 后来各加了0.1 现在改回去了）其他和上面一样 count
dev sentiment acc :: 0.520
dev paraphrase acc :: 0.738
dev sts corr :: 0.730

同上 把count该硬处理1/2
Unsupervised SimCSE train acc :: 0.635
dev sentiment acc :: 0.496
dev paraphrase acc :: 0.662
dev sts corr :: 0.727

同上 把1/2换成 fake random
Unsupervised SimCSE train acc :: 0.623
dev sentiment acc :: 0.518
dev paraphrase acc :: 0.693
dev sts corr :: 0.704

so, 0+0.1 + fake random dropout is better

只保留fake random, 用普通dense layer
train acc :: 0.696
dev sentiment acc :: 0.502
dev paraphrase acc :: 0.783
dev sts corr :: 0.779

so, simple dense layer is better

用普通dense layer, 用普通dropout(+0)
Unsupervised SimCSE train acc :: 0.705
dev sentiment acc :: 0.516
dev paraphrase acc :: 0.761
dev sts corr :: 0.752

用普通dense layer, + 1/2
Unsupervised SimCSE train acc :: 0.699/703
dev sentiment acc :: 0.512
dev paraphrase acc :: 0.754
dev sts corr :: 0.756
dev sentiment acc :: 0.510
dev paraphrase acc :: 0.754
dev sts corr :: 0.746

用普通dense layer, + count
Unsupervised SimCSE train acc :: 0.703
dev sentiment acc :: 0.506
dev paraphrase acc :: 0.749
dev sts corr :: 0.751

在fake random的基础上, 删去了所有cblock
dev sentiment acc :: 0.500
dev paraphrase acc :: 0.741
dev sts corr :: 0.718

so, some cb in classifier is better

还原到最开始(很多emb1 emb2 分开dropout) 删掉forward里面的所有内容 -> 预期会在整体上降低表现(因为unsup不会用在所有任务), 但是可能会提高unsup的表现.
+ 换回普通optimizer
dev sentiment acc :: 0.526
dev paraphrase acc :: 0.765
dev sts corr :: 0.770

**普通优化器, 把伪随机dropout和dense放进forward里面**
dev sentiment acc :: 0.519
dev paraphrase acc :: 0.780
dev sts corr :: 0.752

+ normal optim, 把换成普通单层dropout
dev sentiment acc :: 0.520
dev paraphrase acc :: 0.733
dev sts corr :: 0.727

+ PCG(有weights) 伪随机Dropout在forward里面 有普通量的CB
dev sentiment acc :: 0.470
dev paraphrase acc :: 0.684
dev sts corr :: 0.705

pc_adam.pc_backward([sst_loss, para_loss, sts_loss]) 去掉了weights
dev sentiment acc :: 0.464
dev paraphrase acc :: 0.683
dev sts corr :: 0.727

普通优化器, 伪随机(+0+0), sim只有普通的dense层, 其他CB照旧还是有
dev sentiment acc :: 0.491
dev paraphrase acc :: 0.766
dev sts corr :: 0.754

**普通优化器, 把伪随机dropout(-0.05+0.15)和dense放进forward里面**
dev sentiment acc :: 0.498
dev paraphrase acc :: 0.777
dev sts corr :: 0.760

**普通优化器, 把伪随机dropout(+0+0.1)和dense放进forward里面**
dev sentiment acc :: 0.499
dev paraphrase acc :: 0.774
dev sts corr :: 0.777

**普通优化器, 把伪随机dropout(-0.05+0.1)和dense放进forward里面**
dev sentiment acc :: 0.507
dev paraphrase acc :: 0.729
dev sts corr :: 0.750

**普通优化器, 把伪随机dropout(+0+0.05)和dense放进forward里面**
dev sentiment acc :: 0.488
dev paraphrase acc :: 0.772
dev sts corr :: 0.776

+0+0.15
dev sentiment acc :: 0.506
dev paraphrase acc :: 0.732
dev sts corr :: 0.749

+0+1(平行测试)
dev sentiment acc :: 0.499
dev paraphrase acc :: 0.774
dev sts corr :: 0.777


+ sup simcse phrase + -0.05+0.15 do
dev sentiment acc :: 0.486
dev paraphrase acc :: 0.757
dev sts corr :: 0.788

+ sup simcse phrase +0.0+0.1 do
dev sentiment acc :: 0.498
dev paraphrase acc :: 0.779
dev sts corr :: 0.796

试了去掉tau 是真不行

试了保留tau 然后supsim里面的weight改成1.5
dev sentiment acc :: 0.491
dev paraphrase acc :: 0.749
dev sts corr :: 0.793

weight改成0.5
dev sentiment acc :: 0.499
dev paraphrase acc :: 0.752
dev sts corr :: 0.800

0.1
dev sentiment acc :: 0.500
dev paraphrase acc :: 0.757
dev sts corr :: 0.801

0.05
dev sentiment acc :: 0.497
dev paraphrase acc :: 0.759
dev sts corr :: 0.802

2
dev sentiment acc :: 0.500
dev paraphrase acc :: 0.745
dev sts corr :: 0.791

0.01
dev sentiment acc :: 0.487
dev paraphrase acc :: 0.763
dev sts corr :: 0.803


python -u multitask_classifier.py --option finetune --epochs 2 --batch_size 32 --hidden_dropout_prob 0.2 --lr 3e-05 --use_gpu

1024 hidden, 每个layer123加了一个linear


测试(没用un-sim, 只用sup-sim + multi-task finetune)teacher-student sts
dev sentiment acc :: 0.505
dev paraphrase acc :: 0.708
dev sts corr :: 0.799
对照组
dev sentiment acc :: 0.500
dev paraphrase acc :: 0.798
dev sts corr :: 0.777


对照组(啥都一样,啥都没加,就一个多任务学习)
dev sentiment acc :: 0.504
dev paraphrase acc :: 0.764
dev sts corr :: 0.709
测试FP(用了)
dev sentiment acc :: 0.507
dev paraphrase acc :: 0.742
dev sts corr :: 0.658
测试MS
dev sentiment acc :: 0.521
dev paraphrase acc :: 0.784
dev sts corr :: 0.658

改了similarity的计算(模仿para使用了classifier, 用了sentence-bert的方式)
dev sentiment acc :: 0.489
dev paraphrase acc :: 0.774
dev sts corr :: 0.793
对照组
dev sentiment acc :: 0.499
dev paraphrase acc :: 0.776
dev sts corr :: 0.704
改了similarity的计算, 还把CLS换成 mean pooling
dev sentiment acc :: 0.502
dev paraphrase acc :: 0.770
dev sts corr :: 0.801

```

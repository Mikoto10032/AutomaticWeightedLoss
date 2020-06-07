# AutomaticWeightedLoss

A PyTorch implementation of Liebel L, Körner M. [Auxiliary tasks in multi-task learning](https://arxiv.org/pdf/1805.06334)[J]. arXiv preprint arXiv:1805.06334, 2018. 

The above paper improves the paper "[Multi-task learning using uncertainty to weigh losses for scene geometry and semantics](http://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)" to avoid the loss of becoming negative during training.

## Requirements

* Python
* PyTorch

## How to Train with Your Model

* Clone the repository

``` bash
git clone git@github.com:Mikoto10032/AutomaticWeightedLoss.git
```

* Create an AutomaticWeightedLoss module

```python
from AutomaticWeightedLoss import AutomaticWeightedLoss

awl = AutomaticWeightedLoss(2)
loss1 = 1
loss2 = 2
loss_sum = awl(loss1, loss2)
```

* Create an optimizer to learn weight coefficients

```python
from torch import optim

model = Model()
optimizer = optim.Adam([
                {'params': net.parameters()},
                {'params': awl.parameters()}
            ])
```

## Something to Say

Actually, it is not always effective, but I hope it can help you.
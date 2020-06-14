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

awl = AutomaticWeightedLoss(2)	# we have 2 losses
loss1 = 1
loss2 = 2
loss_sum = awl(loss1, loss2)
```

* Create an optimizer to learn weight coefficients

```python
from torch import optim

model = Model()
optimizer = optim.Adam([
                {'params': model.parameters()},
                {'params': awl.parameters(), 'weight_decay': 0}	
            ])
```

* A complete example

```python
from torch import optim
from AutomaticWeightedLoss import AutomaticWeightedLoss

model = Model()

awl = AutomaticWeightedLoss(2)	# we have 2 losses
loss_1 = ...
loss_2 = ...

# learnable parameters
optimizer = optim.Adam([
                {'params': model.parameters()},
                {'params': awl.parameters()}
            ])

for i in range(epoch):
    for data, label1, label2 in data_loader:
        # forward
        pred1, pred2 = Model(data)	
        # calculate losses
        loss1 = loss_1(pred1, label1)
        loss2 = loes_2(pred2, label2)
        # weigh losses
        loss_sum = awl(loss1, loss2)
        # backward
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
```

## Something to Say

Actually, it is not always effective, but I hope it can help you.
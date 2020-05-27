# Transformer Pointer Coreference Resolution

## Trained Model
### Parameters
The final trained model was trained using the following parameters:
```
Model Hyperparameters
---------------------
Model Hidden Dimension: 512
Num Layers: 4
Num Attention Heads: 8
Dropout: 0.2
Learning Rate: 0.0001 (1e-4)
Final Learning Rate: 6.3e-5
Num Epochs: 10
Batch Size: 4

Meta
    - criterion: nn.CrossEntropyLoss
    - optimizer: nn.SGD
    - scheduler: StepLR(optim, step_size=1, gamma=0.95)
    - src_mask: None
    - tgt_mask: Used
    - torch.manual_seed = 42

Training Data
    - train.tsv
    - TEXT and LABEL vocab
    - NO_REF_TOKEN: '<nr>'
```
This model achieved the following losses:
```
Validation Losses
    - Epoch 1: 416.91417050361633
    - Epoch 2: 323.9406987428665
    - Epoch 3: 225.71645319461823
    - Epoch 4: 144.5937288403511
    - Epoch 5: 91.24655830860138
    - Epoch 6: 61.019713655114174
    - Epoch 7: 44.32578928396106
    - Epoch 8: 34.620120372623205
    - Epoch 9: 28.595584958791733
    - Epoch 10: 24.60211064480245
```

### Training
The above model was trained in Google Colab using the entirety of the training data (~30,000 documents) and validation data (500 documents) on a Tesla P100 GPU over 5 hours.

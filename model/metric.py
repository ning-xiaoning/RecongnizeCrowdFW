import torch
import numpy as np

def MAE(ouput, target):
    with torch.no_grad():
        mae = np.abs(ouput - target)
    return mae 

def MSE(output, target):
    with torch.no_grad():
        mse = (output-target) **2
    return mse
        

# def accuracy(output, target):
#     with torch.no_grad():
#         pred = torch.argmax(output, dim=1)
#         assert pred.shape[0] == len(target)
#         correct = 0
#         correct += torch.sum(pred == target).item()
#     return correct / len(target)


# def top_k_acc(output, target, k=3):
#     with torch.no_grad():
#         pred = torch.topk(output, k, dim=1)[1]
#         assert pred.shape[0] == len(target)
#         correct = 0
#         for i in range(k):
#             correct += torch.sum(pred[:, i] == target).item()
#     return correct / len(target)

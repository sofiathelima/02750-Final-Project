# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 00:12:57 2023
"""
import os
import json
import numpy as np
import pandas as pd

import time

#packages for GNN
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

class GraphSAGE(torch.nn.Module):
	"""GraphSAGE"""
	def __init__(self, dim_h, dim_in, dim_out):
		super().__init__()
		self.sage1 = SAGEConv(dim_in, dim_h)
		self.sage2 = SAGEConv(dim_h, dim_out)
		# self.optimizer = torch.optim.Adam(self.parameters(),
        #                               lr=0.01,
        #                               weight_decay=10e-4)

	# def forward(self, x, edge_index):
	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		h = self.sage1(x, edge_index).relu()
		h = self.sage2(h, edge_index)
		return F.softmax(h, dim=1)
	
		
class GraphSAGE_mc(torch.nn.Module):
	"""GraphSAGE"""
	def __init__(self, dim_h, dim_in, dim_out,
				  mc_prob:float=0.5):
		super().__init__()
		self.sage1 = SAGEConv(dim_in, dim_h)
		self.sage2 = SAGEConv(dim_h, dim_out)
		# self.optimizer = torch.optim.Adam(self.parameters(),
        #                               lr=0.01,
        #                               weight_decay=10e-4)

		self.mc_prob = mc_prob
	
# 		self.dropout = F.dropout(p=self.mc_prob)
		
	# def forward(self, x, edge_index):
	def forward(self, data):
		x, edge_index = data.x, data.edge_index

		h = self.sage1(x, edge_index).relu()
        
		h = F.dropout(h, p=self.mc_prob, training=True)
        
		h = self.sage2(h, edge_index)
		        
		
		return F.softmax(h, dim=1)
	

def construct_graph(save_path, data_num, node_labels,
					edges=None):

	with open(os.path.join(save_path, f'data_{data_num}.json'), 'r') as f:
		graph_dict = json.load(f)

	x = torch.tensor(graph_dict['x'], dtype=torch.float)
	y = torch.tensor(node_labels, dtype=torch.long)
	faces = torch.tensor(graph_dict['face'], dtype=torch.long)
	if edges == None:
		edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long)
		graph = Data(x=x, y=y, edge_index=edge_index, face = faces)		
	else:
		edge_index = torch.tensor(np.array(edges), dtype=torch.long)	
	
		graph = Data(x=x, y=y, edge_index=edge_index.t().contiguous(), face = faces)
	
	return graph


def accuracy(pred_y, y):
	"""Calculate accuracy."""
	return ((pred_y == y).sum() / len(y)).item()
	
# def confusion_matrix_numpy(y_true, y_pred, num_classes):
#     matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
#     for i in range(num_classes):
#         for j in range(num_classes):
#             matrix[i, j] = np.sum((y_true == i) & (y_pred == j))
#     return matrix

def confusion_matrix_torch(y_true, y_pred, num_classes):
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)
    for i in range(num_classes):
        for j in range(num_classes):
            matrix[i, j] = torch.sum((y_true == i) & (y_pred == j))
    return matrix

def dice_score_torch(y_true, y_pred, num_classes):
    cm = confusion_matrix_torch(y_true, y_pred, num_classes)
    TP = torch.diag(cm)
    FP = torch.sum(cm, dim=0) - TP
    FN = torch.sum(cm, dim=1) - TP

    dice_scores = 2 * TP / (2 * TP + FP + FN)
    return dice_scores.mean()

def iou_torch(y_true, y_pred, num_classes):
    cm = confusion_matrix_torch(y_true, y_pred, num_classes)
    TP = torch.diag(cm)
    FP = torch.sum(cm, dim=0) - TP
    FN = torch.sum(cm, dim=1) - TP
    union = TP + FP + FN

    iou_scores = TP / union
    return iou_scores.mean()


# Jaccard Index is also known as IoU
def jaccard_index(pred, labels, smooth=1):
    gt = np.array(labels.cpu())
    seg = np.array(pred.cpu())
    numerator = np.sum(gt * seg) + smooth
    denominator = (np.sum(gt + seg)-numerator) + smooth

    return np.mean(numerator / denominator)

#Calcluate Dice coefficient, dice_coef = mean(2 * intersect / (intersect + union))
def dice_coef(pred, labels, smooth = 1):
    gt = np.array(labels.cpu())
    seg = np.array(pred.cpu())
    numerator = 2 * np.sum(gt * seg) + smooth
    denominator = np.sum(gt + seg) + smooth

    return np.mean(numerator / denominator)

	

def test_model(model, device, loader):
    num_classes = 12
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    loss = 0
    acc = 0
    iou = 0
    dice = 0
    predict = []

    with torch.no_grad():
        for data in loader:
            bs = len(data)
            # predicted = model(data.x.to(device), data.edge_index.to(device))
            predicted = model(data)
            # data = data[0]
	    
            # loss += criterion(predicted, data.y.to(device)) / len(loader)
            y = torch.cat([d.y for d in data]).to(predicted.device)
            loss += criterion(predicted, y) / len(loader)
            labels = y

            preds = predicted.argmax(dim=1)
            # labels = data.y.to(device)
	    
            # acc += accuracy(predicted.argmax(dim=1), data.y.to(device)) / len(loader)
            acc += accuracy(predicted.argmax(dim=1), y) / len(loader)
	    
            dice += dice_score_torch(labels, preds, num_classes) / len(loader)
            iou += iou_torch(labels, preds, num_classes) / len(loader)
            # iou += jaccard_index(predicted.argmax(dim=1), data.y.to(device)) / len(loader)
            # dice += dice_coef(predicted.argmax(dim=1), data.y.to(device)) / len(loader)

            predict.append(predicted.view(bs,-1,12))

    model.train()
    
    # print("predict 0 shape:",predict[0].shape)
    # print("predict 0 unsqueeze shape:",predict[0].unsqueeze(0).shape)
    # print("predict -1 shape:",predict[-1].shape)
    # print("predict -1 unsqueeze shape:",predict[-1].unsqueeze(0).shape)

    return loss, acc, iou, dice, torch.cat(predict,dim=0)

def train_model(model, device, loader, val_loader,	epochs=10):
	num_classes = 12
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(),
                                      lr=0.01,
                                      weight_decay=10e-4)
	# optimizer = model.optimizer
	epochs = epochs
	metrics_df = pd.DataFrame(columns={'loss':[],
					'acc':[],
					'val_loss':[],
					'val_acc':[]})

	time_curr = time.time()
	model.train()
	for epoch in range(epochs+1):
		
		total_loss = 0
		acc = 0
		iou = 0
		dice = 0
		total_val_loss = 0
		total_val_acc = 0
		total_val_iou = 0
		total_val_dice = 0

		# Train on batches
		for i, data in enumerate(loader):
			# print(len(data),data)
			# print(type(data),data)
			optimizer.zero_grad()
			# predicted = model(data.x.to(device), data.edge_index.to(device))
			predicted = model(data)

			y = torch.cat([d.y for d in data]).to(predicted.device)
			loss = criterion(predicted, y) / len(loader)
			labels = y

			# loss = criterion(predicted, data.y.to(device))

			preds = predicted.argmax(dim=1)
			# labels = data.y.to(device)
			
			
			#Record losses
			total_loss += criterion(predicted, y) / len(loader)
			acc += accuracy(predicted.argmax(dim=1), y) / len(loader)	

			# total_loss += criterion(predicted, data.y.to(device)) / len(loader)
			# acc += accuracy(predicted.argmax(dim=1), data.y.to(device)) / len(loader)	

			dice += dice_score_torch(labels, preds, num_classes) / len(loader)
			iou += iou_torch(labels, preds, num_classes) / len(loader)
			# iou += jaccard_index(predicted.argmax(dim=1), data.y.to(device)) / len(loader)
			# dice += dice_coef(predicted.argmax(dim=1), data.y.to(device)) / len(loader)
			
			loss.backward()
			optimizer.step()

			# Validation
			val_loss, val_acc, val_iou, val_dice, _ = test_model(model, device, val_loader)
			total_val_loss += val_loss
			total_val_acc += val_acc				  		  			  	  
			total_val_iou += val_iou
			total_val_dice += val_dice

		# Print metrics every 10 epochs
		if(epoch % 1 == 0):
					print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
						f'| Train Acc: {acc*100:>5.2f}% '
						f'| Val Loss: {val_loss:.2f} '
						f'| Val Acc: {val_acc*100:.2f}%'
						f'| Val IoU: {val_iou*100:.2f}%'
						f'| Val DICE: {val_dice*100:.2f}%'
						f'| Time: {int(time.time() - time_curr)}')
					time_curr = time.time()
				
	return val_loss, val_acc, val_iou, val_dice, model
	
def save_model(model, save_path):		
    torch.save(model.state_dict(), save_path)
	    


		
		



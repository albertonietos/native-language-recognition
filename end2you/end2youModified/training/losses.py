import torch
import torch.nn as nn

from functools import partial


class Losses:
    
    def __init__(self,
                 loss:str = 'mse'):
        self._loss = self._get_loss(loss)
        self.loss_fn = self.masked_loss
        self.loss_name = loss
    
    def _get_loss(self, loss):
        return {
            'mse': nn.MSELoss(),
            'ccc': self.ccc,
            'ce': partial(self.cross_entropy_loss, nn.CrossEntropyLoss())
        }[loss]
    def masked_loss(self, predictions, labels, mask):

        """
        num_samples = len(mask)
        batch_loss = 0.0
        #print(mask)
        for i in range(num_samples):
            m = mask[i]
            #print(m)
            #batch_loss += self._loss(
            #    predictions[i,:m].view(-1), labels[i,:m].view(-1))
            batch_loss += self._loss(
               predictions[i,:m].view(-1), labels[i].view(-1))
        
        """
        #print(mask)
        num_samples = len(mask)
        #print(labels)
        labels = torch.argmax(labels,dim=2)
        #print("XD")
        #print(labels)

        batch_loss = 0.0
        for i in range(num_samples):
            batch_loss += self._loss(predictions[i,:,:], labels[i,:])
        
        return batch_loss/num_samples
    
    def ccc(self, predictions, labels):
        
        def _get_moments(data_tensor):
            mean_t = torch.mean(data_tensor, 0)
            var_t = torch.var(data_tensor, 0, unbiased=False)
            return mean_t, var_t
        
        labels_mean, labels_var = _get_moments(labels)
        pred_mean, pred_var = _get_moments(predictions)
        
        mean_cent_prod = torch.mean((predictions - pred_mean) * (labels - labels_mean))
        batch_ccc = 1 - (2 * mean_cent_prod) / (pred_var + labels_var + torch.square(pred_mean - labels_mean))

        return torch.mean(batch_ccc)
    
    def cross_entropy_loss(self, instance_cross_entropy_loss, predictions, labels):
        
        #predictions = predictions.unsqueeze(1)
        #print(predictions)
        
        labels = labels.type(torch.long)
        #print(labels)
        #print(instance_cross_entropy_loss(predictions, labels, False))
        return instance_cross_entropy_loss(predictions, labels)

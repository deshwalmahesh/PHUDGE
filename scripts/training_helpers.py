import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling

from sklearn.metrics import (r2_score, 
                             explained_variance_score,
                             mean_squared_error,
                             mean_absolute_percentage_error, 
                             mean_absolute_error,
                            accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)


import numpy as np
import random, os, warnings
import wandb
import scipy.stats
from typing import Any, Dict, List, Optional, Tuple, Union



class CustomMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def compute_metrics_for_scores(self, inputs):
        preds, labels = inputs
        
        if self.num_classes == 1:
            preds = preds.squeeze()
            acc = accuracy_score(labels.astype(int), np.round(preds, decimals=0).astype(int))
        else:
            preds = np.argmax(preds, axis=1)
            acc = accuracy_score(labels, preds)

        mse = mean_squared_error(labels, preds)
        rmse = mean_squared_error(labels, preds, squared=False)
        mae = mean_absolute_error(labels, preds)
        r2 = r2_score(labels, preds)

        pearson_r = scipy.stats.pearsonr(labels.squeeze(), preds).statistic
        spearman_rho = scipy.stats.spearmanr(labels, preds).statistic
        kendall_tau = scipy.stats.kendalltau(labels, preds).statistic

        return {"accuracy": acc, "r2": r2, "mse": mse, "rmse": rmse, "mae": mae, 
               "pearson_r" : pearson_r, "spearman_rho" : spearman_rho, "kendall_tau" : kendall_tau}
    

def seed_everything(seed=13):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
        
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



class CustomDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
               
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                   
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        return batch



class CustomEMDLoss(nn.Module):
    """
    If the actual Class/Label/Score was "0" and model predicted class "4", then you want it to penalise more than if it would have predicted it class "2"
    """

    def __init__(self, num_classes:int, device:str, p = 2, alpha = 2, squared = True, summed = False, with_sanity_check = True):
        """
        args:
            num_classes: >= 2. Number of labels / classes in your Classification task
            p: The norm (2 means L-2) norm to use
            alpha: A more generalised form. If r == alpha == 2, it behaves as the squared summed else it'll penalise the mis-classification more if alpha < p
            squared: Whether to compute (L1,2,..) Normalized Vanilla or Squared EMD loss as given in paper: https://arxiv.org/pdf/1611.05916.pdf
            summed: Whether to add the loss for all the samples or take mean of whole batch
            sanity_check: If sanity check, it'll be slowe but will look whether Logits are passed or softmaxed. Also will look for the Labels are One Hot encoded or not. Will do both internally. Note: It'll be a bit slower
        """
        super(CustomEMDLoss, self).__init__()
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.squared = squared
        self.summed = summed
        self.sanity_check = with_sanity_check
        self.device = device

    
    def is_softmaxed(self, tensor):
        """
        Just a small function to see if we have Logits or Softmaxed Probs
        """
        result = (tensor >= 0).all() and (tensor <= 1).all() and torch.isclose(tensor.sum(), torch.tensor(1.0).to(self.device))
        # if not result: print("is NOT Softmaxed")
        return result
    
    def is_one_hot(self, tensor):
        """
        Small function to Sanity check on the One Hot encodings
        """
        if tensor.dim() != 2: return False  # Check if the tensor is 2D
       
        row_sums = tensor.sum(dim=1)
        if not torch.all(row_sums == 1): return False  # Check if all rows sum to 1

        unique_values = tensor.unique()
        if not torch.all((unique_values == 0) | (unique_values == 1)):
            print("Is NOT One Hot Encoded")
            return False  # Check if all values are 0 or 1
    
        return True


    def forward(self, logits, labels):
        """
        Args:
            logits: raw output (logits) for predicted labels of shape [BATCH × num_classes]
            labels: Actual labels of shape [BATCH × num_classes] or [BATCH]. If it's a Batch, then it'll be One hot encoded to [BATCH × num_classes]
        """
        if self.sanity_check:
            if len(logits) > 1: labels = labels.squeeze() # Handle Batch of size 1
            if not self.is_one_hot(labels): labels = torch.nn.functional.one_hot(labels.long(), num_classes=self.num_classes)
            
            if not self.is_softmaxed(logits): logits = torch.nn.functional.softmax(logits, dim=1)

        assert logits.shape == labels.shape, f"Shape of the two distribution batches must be the same. Got {logits.shape} and {labels.shape}"

        cum_logits = torch.cumsum(logits, dim=1)
        cum_labels = torch.cumsum(labels, dim=1)

        if self.squared:
            emd = torch.square(cum_labels - cum_logits)
            if self.summed: emd = emd.sum(axis = 1)
        
        else: emd = torch.linalg.norm(cum_labels - cum_logits, ord=self.p, dim=1) ** (self.alpha) # When alpha == p == 2, it is equal to the above form

        return emd.mean()

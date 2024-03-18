"""
File to train the GNN model.
"""

import os
from pathlib import Path
import torch
import shutil
import warnings
import numpy as np
import json
from torch.optim import Adam
from utils.parser_utils import (
    arg_parse,
    create_args_group,
    fix_random_seed,
    get_data_args,
    get_graph_size_args,
)
from utils.io_utils import check_dir
from gendata import get_dataloader, get_dataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from gnn.model import get_gnnNets
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score


# Save directory model_name + dataset_name + layers + hidden_dim

class TrainModel(object):
    def __init__(
        self,
        model,
        dataset,
        device,
        task="regression",
        task_target="graph",
        save_dir=None,
        save_name="model",
        **kwargs,
    ):
        self.model = model
        self.dataset = dataset  # train_mask, eval_mask, test_mask
        self.loader = None
        self.device = device
        self.task = task
        self.task_target = task_target
        self.optimizer = None
        self.save = save_dir is not None
        self.save_dir = save_dir
        self.save_name = save_name
        check_dir(self.save_dir)

        if self.task_target=="graph":
            dataloader_params = kwargs.get("dataloader_params")
            self.loader,_,_,_ = get_dataloader(dataset, **dataloader_params)

    def __loss__(self, logits, labels):
        if self.task.endswith("classification"):
            return F.nll_loss(logits, labels)
        elif self.task == "regression":
            return F.mse_loss(logits, labels)

    # Get the loss, apply optimizer, backprop and return the loss

    def _train_batch(self, data, labels):
        logits = self.model(data=data)
        if self.task.endswith("classification"):
            loss = self.__loss__(logits, labels)
        elif self.task == "regression":
            loss = self.__loss__(logits, labels)
        else:
            loss = self.__loss__(logits[data.train_mask], labels[data.train_mask])           

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=2.0)
        self.optimizer.step()
        return loss.item()

    def _eval_batch(self, data, labels, **kwargs):
        self.model.eval()
        logits = self.model(data)
        if self.task.endswith("classification"):
            loss = self.__loss__(logits, labels)
            loss = loss.item()
            preds = logits.argmax(-1)
        elif self.task == "regression":
            loss = self.__loss__(logits, labels)
            loss = loss.item()
            preds = logits
        else:
            mask = kwargs.get("mask")
            if mask is None:
                warnings.warn("The node mask is None")
                mask = torch.ones(labels.shape[0])
            loss = self.__loss__(logits[mask], labels[mask])
            loss = loss.item()
            preds = logits.argmax(-1)

        return loss, preds

    def eval(self):
        self.model.to(self.device)
        self.model.eval()

        if self.task.endswith("classification"):
            losses, accs, balanced_accs, f1_scores = [], [], [], []
            for batch in self.loader["eval"]:
                batch = batch.to(self.device)
                loss, batch_preds = self._eval_batch(batch, batch.y)
                losses.append(loss)
                accs.append(batch_preds == batch.y)
                balanced_accs.append(balanced_accuracy_score(batch.y.cpu(), batch_preds.cpu()))
                f1_scores.append(f1_score(batch.y.cpu(), batch_preds.cpu(), average="weighted"))
            eval_loss = torch.tensor(losses).mean().item()
            eval_acc = torch.cat(accs, dim=-1).float().mean().item()
            eval_balanced_acc = np.mean(balanced_accs)
            eval_f1_score = np.mean(f1_scores)
            print(
                f"Test loss: {eval_loss:.4f}, test acc {eval_acc:.4f}, balanced test acc {eval_balanced_acc:.4f}, test f1 score {eval_f1_score:.4f}"
            )
            return eval_loss, eval_acc, eval_balanced_acc, eval_f1_score
        elif self.task == "regression":
            losses, r2scores = [], []
            for batch in self.loader["eval"]:
                batch = batch.to(self.device)
                loss, batch_preds = self._eval_batch(batch, batch.y)
                r2scores.append(r2_score(batch.y.cpu(), batch_preds.cpu()))
                losses.append(loss)
            eval_loss = torch.tensor(losses).mean().item()
            eval_r2score = np.mean(r2scores)
            print(
                f"eval loss: {eval_loss:.6f}, eval r2score {eval_r2score:.6f}"
            )
            return eval_loss, eval_r2score
        else:
            data = self.dataset.data.to(self.device)
            eval_loss, preds = self._eval_batch(data, data.y, mask=data.val_mask)
            eval_acc = (preds == data.y).float().mean().item()
            eval_balanced_acc = balanced_accuracy_score(data.y, preds)
            eval_f1_score = f1_score(data.y, preds, average="weighted")
        return eval_loss, eval_acc, eval_balanced_acc, eval_f1_score


    def test(self):
        state_dict = torch.load(
            os.path.join(self.save_dir, f"{self.save_name}_latest.pth")
        )["net"]
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.task.endswith("classification"):
            losses, preds, accs, balanced_accs, f1_scores = [], [], [], [], []
            for batch in self.loader["test"]:
                batch = batch.to(self.device)
                loss, batch_preds = self._eval_batch(batch, batch.y)
                losses.append(loss)
                preds.append(batch_preds)
                accs.append(batch_preds == batch.y)
                balanced_accs.append(balanced_accuracy_score(batch.y.cpu(), batch_preds.cpu()))
                f1_scores.append(f1_score(batch.y.cpu(), batch_preds.cpu(), average="weighted"))
            test_loss = torch.tensor(losses).mean().item()
            preds = torch.cat(preds, dim=-1)
            test_acc = torch.cat(accs, dim=-1).float().mean().item()
            test_balanced_acc = np.mean(balanced_accs)
            test_f1_score = np.mean(f1_scores)
            print(
                f"Test loss: {test_loss:.4f}, test acc {test_acc:.4f}, balanced test acc {test_balanced_acc:.4f}, test f1 score {test_f1_score:.4f}"
            )
            scores = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_balanced_acc": test_balanced_acc,
            "test_f1_score": test_f1_score,
            }
            self.save_scores(scores)
            return test_loss, test_acc, test_balanced_acc, test_f1_score, preds
        elif self.task == "regression":
            losses, r2scores, preds = [], [], []
            for batch in self.loader["test"]:
                batch = batch.to(self.device)
                loss, batch_preds = self._eval_batch(batch, batch.y)
                preds.append(batch_preds)
                r2scores.append(r2_score(batch.y.detach().cpu(), batch_preds.detach().cpu()))
                losses.append(loss)
            test_loss = torch.tensor(losses).mean().item()
            test_r2score = np.mean(r2scores)
            preds = torch.cat(preds, dim=-1)
            print(
                f"test loss: {test_loss:.6f}, test r2score {test_r2score:.6f}"
            )
            scores = {
            "test_loss": test_loss,
            "test r2score": test_r2score,
            }
            self.save_scores(scores)
            return test_loss, test_r2score, preds
        else:
            data = self.dataset.data.to(self.device)
            test_loss, preds = self._eval_batch(data, data.y, mask=data.test_mask)
            test_acc = (preds == data.y).float().mean().item()
            test_balanced_acc = balanced_accuracy_score(data.y, preds)
            test_f1_score = f1_score(data.y, preds, average="weighted")
            print(
                f"Test loss: {test_loss:.4f}, test acc {test_acc:.4f}, balanced test acc {test_balanced_acc:.4f}, test f1 score {test_f1_score:.4f}"
            )
            scores = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_balanced_acc": test_balanced_acc,
            "test_f1_score": test_f1_score,
            }
            self.save_scores(scores)
            return test_loss, test_acc, test_balanced_acc, test_f1_score, preds

    # Train model
    def train(self, train_params=None, optimizer_params=None):
        if self.task.endswith("classification"):
            num_epochs = train_params["num_epochs"]
            num_early_stop = train_params["num_early_stop"]
            #milestones = train_params["milestones"] # needed if using a different LR scheduler
            #gamma = train_params["gamma"]
            
            if optimizer_params is None:
                self.optimizer = Adam(self.model.parameters())
            else:
                self.optimizer = Adam(self.model.parameters(), **optimizer_params)
            
            lr_schedule = ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.2, patience=2, min_lr=0.001
            )
            
            self.model.to(self.device)
            best_eval_acc = 0.0
            best_eval_loss = 0.0
            early_stop_counter = 0
            for epoch in range(num_epochs):
                is_best = False
                self.model.train()
                if self.task.endswith("classification"):
                    losses = []
                    for batch in self.loader["train"]:
                        batch = batch.to(self.device)
                        loss = self._train_batch(batch, batch.y)
                        losses.append(loss)
                    train_loss = torch.FloatTensor(losses).mean().item()
            
                else:
                    data = self.dataset.data.to(self.device)
                    train_loss = self._train_batch(data, data.y)
            
                with torch.no_grad():
                    eval_loss, eval_acc, eval_balanced_acc, eval_f1_score = self.eval()
            
                print(
                    f"Epoch:{epoch}, Training_loss:{train_loss:.4f}, Eval_loss:{eval_loss:.4f}, Eval_acc:{eval_acc:.4f}, Eval_balanced_acc:{eval_balanced_acc:.4f}, Eval_f1_score:{eval_f1_score:.4f}, lr:{self.optimizer.param_groups[0]['lr']}"
                )
                if num_early_stop > 0:
                    if eval_loss <= best_eval_loss:
                        best_eval_loss = eval_loss
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                    if epoch > num_epochs / 2 and early_stop_counter > num_early_stop:
                        break
                if lr_schedule:
                    lr_schedule.step(eval_acc)
            
                if best_eval_acc < eval_acc:
                    is_best = True
                    best_eval_acc = eval_acc
                recording = {"epoch": epoch, "is_best": str(is_best)}
                if self.save:
                    self.save_model(is_best, recording=recording)
        
        elif self.task == "regression":
            num_epochs = train_params["num_epochs"]
            num_early_stop = train_params["num_early_stop"]
            # milestones = train_params["milestones"] # needed if using a different LR scheduler
            # gamma = train_params["gamma"]

            if optimizer_params is None:
                self.optimizer = Adam(self.model.parameters())
            else:
                self.optimizer = Adam(self.model.parameters(), **optimizer_params)

            lr_schedule = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.1, patience=2
            )

            self.model.to(self.device)
            best_eval_r2score = -100.0
            best_eval_loss = 0.0
            early_stop_counter = 0
            for epoch in range(num_epochs):
                is_best = False
                self.model.train()
                if self.task == "regression":
                    losses = []
                    for batch in self.loader["train"]:
                        batch = batch.to(self.device)
                        loss = self._train_batch(batch, batch.y)
                        losses.append(loss)
                    train_loss = torch.FloatTensor(losses).mean().item()

                else:
                    data = self.dataset.data.to(self.device)
                    train_loss = self._train_batch(data, data.y)

                with torch.no_grad():
                    eval_loss, eval_r2score = self.eval()

                print(
                    f"Epoch:{epoch}, Training_loss:{train_loss:.4f}, Eval_loss:{eval_loss:.4f}, Eval_r2:{eval_r2score:.4f}, lr:{self.optimizer.param_groups[0]['lr']}"
                )
                if num_early_stop > 0:
                    if eval_loss <= best_eval_loss:
                        best_eval_loss = eval_loss
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                    if epoch > num_epochs / 2 and early_stop_counter > num_early_stop:
                        break
                if lr_schedule:
                    lr_schedule.step(eval_loss)

                if best_eval_r2score < eval_r2score:
                    is_best = True
                    best_eval_r2score = eval_r2score
                recording = {"epoch": epoch, "is_best": str(is_best)}
                if self.save:
                    self.save_model(is_best, recording=recording)

    # Save each latest and best model
    def save_model(self, is_best=False, recording=None):
        self.model.to("cpu")
        state = {"net": self.model.state_dict()}
        for key, value in recording.items():
            state[key] = value
        latest_pth_name = f"{self.save_name}_latest.pth"
        best_pth_name = f"{self.save_name}_best.pth"
        ckpt_path = os.path.join(self.save_dir, latest_pth_name)
        torch.save(state, ckpt_path)
        if is_best:
            print("saving best...")
            shutil.copy(ckpt_path, os.path.join(self.save_dir, best_pth_name))
        self.model.to(self.device)

    # Load the best model
    def load_model(self):
        state_dict = torch.load(
            os.path.join(self.save_dir, f"{self.save_name}_best.pth")
        )["net"]
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def save_scores(self, scores):
        with open(os.path.join(self.save_dir, f"{self.save_name}_scores.json"), "w") as f:
            json.dump(scores, f)

#  Main train function
def train_gnn(args, args_group):
    fix_random_seed(args.seed)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    dataset_params = args_group["dataset_params"]
    model_params = args_group["model_params"]

    # changing the dataset path here, load the dataset
    dataset = get_dataset(
        dataset_root=os.path.join(args.data_save_dir, args.dataset_name),
        **dataset_params,
    )
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    # get dataset args
    args = get_data_args(dataset, args)

    if args.task.endswith("classification") | (args.task == "regression"):
        dataloader_params = {
            "batch_size": args.batch_size,
            "random_split_flag": eval(args.random_split_flag),
            "data_split_ratio": [args.train_ratio, args.val_ratio, args.test_ratio],
            "seed": args.seed,
        }
    # get model
    model = get_gnnNets(args.num_node_features, args.num_classes, model_params, args.task)

    # train model
    trainer = TrainModel(
        model=model,
        dataset=dataset,
        device=device,
        task=args.task,
        task_target = args.task_target,
        save_dir=os.path.join(args.model_save_dir, args.dataset_name, args.task_target),
        save_name=f"{args.model_name}_{args.datatype}_{args.num_layers}l_{args.hidden_dim}h",
        dataloader_params=dataloader_params,
    ) 

    if Path(os.path.join(trainer.save_dir, f"{trainer.save_name}_best.pth")).is_file():
        trainer.load_model()
    else:
        trainer.train(
            train_params=args_group["train_params"],
            optimizer_params=args_group["optimizer_params"],
        )
    # test model
    if args.task == "regression":
        _, _, _ = trainer.test()
    else:
        _, _, _, _, _ = trainer.test()


if __name__ == "__main__":
    parser, args = arg_parse()
    args = get_graph_size_args(args)

    # for loop the training architecture for the number of layers and hidden dimensions
    for j in [8, 16, 32]:   # hidden dimension
        for i in [1, 2, 3]:  # number of layers
            args.num_layers = i
            args.hidden_dim = j

            args_group = create_args_group(parser, args)
            print(f"Hidden_dim: {args.hidden_dim}, Num_layers: {args.num_layers}")
            train_gnn(args, args_group)

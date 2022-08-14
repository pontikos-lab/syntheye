# import libraries
import os
import time
import torch
from tqdm import tqdm
from clutils import History, logger
from typing import Any

class Trainer():
    """ A simple interface for implementing classifier training """

    def __init__(self, model: Any, device: Any = None):
        self.device = device
        self.model = model
        self.history = History()

    def configure_devices(self) -> None:

        # push model to device (Default cpu)
        if isinstance(self.device, list):
            self.device = [torch.device(f"cuda:{d}") for d in self.device]
            self.model.to(self.device[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device)
        
        elif isinstance(self.device, int):
            self.model.to(torch.device(f"cuda:{self.device}"))

        else:
            self.model.to("cpu")

        return None

    def save_weights(self, monitor : str = "loss", save_to : str = "") -> None:
        """ Save weights """

        if monitor == "loss":
            if self.epoch_cur_loss < self.best_value:
                logger("Loss decreased from {} to {}, updating model weights\n".format(self.best_value, self.epoch_cur_loss))
                self.best_value = self.epoch_cur_loss
                if isinstance(self.model, torch.nn.DataParallel):
                    torch.save(self.model.module.state_dict(), save_to)
                else:
                    torch.save(self.model.state_dict(), save_to)
            else:
                logger("Current loss is not smaller than the best loss, no weight update.\n")
        
        elif monitor == "acc":
            if self.epoch_cur_acc > self.best_value:
                logger("Accuracy increased from {} to {}, updating model weights\n".format(best_val, self.epoch_cur_acc))
                best_val = self.epoch_cur_acc
                if isinstance(self.model, torch.nn.DataParallel):
                    torch.save(self.model.module.state_dict(), save_to)
                else:
                    torch.save(self.model.state_dict(), save_to)
            else:
                logger("Current accuracy is not larger than the best accuracy, no weight update.\n")
        
        else:
            raise ValueError("Can only monitor be `loss` or `acc`.")

        return None

    def train(self, dataloaders: Any, loss_fn: Any, optimizer: Any, num_epochs: int, save_dir: str, save_best_weights: bool, monitor: str, **kwargs) -> None:

        # configure devices
        self.configure_devices()

        # store dataset sizes
        sizes = {"Train": len(dataloaders["Train"].dataset), "Val": len(dataloaders["Val"].dataset)}

        # track loss and accuracy - for saving best weights
        self.epoch_cur_loss, self.epoch_cur_acc = float("inf"), 0.0
        self.best_value = float("inf") if monitor == "loss" else 0.0
        
        # saves training history
        since = time.time()

        # begin training
        for epoch in range(num_epochs):

            logger(f"Epoch {epoch+1}/{num_epochs}")

            for phase in ["Train", "Val"]:
                
                # switch between a training/inference mode
                if phase == "Train":
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                # run batch training
                for _, _, inputs, labels in tqdm(dataloaders[phase]):

                    # push inputs to device
                    try:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    except:
                        inputs = inputs.to(self.device[0])
                        labels = labels.to(self.device[0])

                    # zero gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'Train'):
                        if ("is_inception" in kwargs) and (phase == "Train"):
                            outputs, aux_outputs = self.model(inputs)
                            loss1 = loss_fn(outputs, labels)
                            loss2 = loss_fn(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = self.model(inputs)
                            loss = loss_fn(outputs, labels)

                        preds = torch.argmax(torch.nn.functional.softmax(outputs, dim=1), dim=1)
                        # backward + optimize only if in training phase
                        if phase == 'Train':
                            loss.backward()
                            optimizer.step()

                    # statistics per batch
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                # average statistics over entire dataset
                self.epoch_cur_loss = running_loss / sizes[phase]
                self.epoch_cur_acc = running_corrects.double() / sizes[phase]

                # print loss and acc to terminal
                logger("{} Loss: {:.4f} Acc: {:.4f}".format(phase, self.epoch_cur_loss, self.epoch_cur_acc))

                # update history dictionary
                self.history.add_entry(phase, self.epoch_cur_loss, self.epoch_cur_acc.item())

                if (phase == "Val"):
                    if save_best_weights:
                        self.save_weights(monitor=monitor, save_to=os.path.join(save_dir, "best_weights.pth"))
            
            # save latest results
            self.history.dump(os.path.join(save_dir, "training_logs.json"))
            self.history.plot(save_dir)

        time_elapsed = time.time() - since

        # print train summary
        logger("Training Summary")
        logger("----------------")
        logger(f"Total Time Elapsed = {time_elapsed//60}m {time_elapsed % 60}s")
        logger(f"Saved weights to: " + save_dir + "best_weights.pth")
        
        return None
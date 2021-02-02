import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter

from xutils.dl.pytorch.utils import get_device


def compute_metrics(model,
                    test_loader,
                    plot_roc_curve=False,
                    device=None,
                    positive_label="Positive",
                    negative_label="Negative",
                    verbose=True):
    if device is None:
        device = get_device()

    model.eval()

    val_loss = 0
    val_correct = 0

    criterion = nn.CrossEntropyLoss()

    score_list = torch.Tensor([]).to(device)
    pred_list = torch.Tensor([]).to(device).long()
    target_list = torch.Tensor([]).to(device).long()
    path_list = []

    for iter_num, data in enumerate(test_loader):
        # Convert image data into single channel data
        image, target = data['img'].to(device), data['label'].to(device)
        paths = data['paths']
        path_list.extend(paths)

        # Compute the loss
        with torch.no_grad():
            output = model(image)

        # Log loss
        val_loss += criterion(output, target.long()).item()

        # Calculate the number of correctly classified examples
        pred = output.argmax(dim=1, keepdim=True)
        val_correct += pred.eq(target.long().view_as(pred)).sum().item()

        # Bookkeeping
        score_list = torch.cat([score_list, nn.Softmax(dim=1)(output)[:, 1].squeeze()])
        pred_list = torch.cat([pred_list, pred.squeeze()])
        target_list = torch.cat([target_list, target.squeeze()])

    classification_metrics = classification_report(target_list.tolist(), pred_list.tolist(),
                                                   target_names=[negative_label, negative_label],
                                                   output_dict=True)

    sensitivity = classification_metrics[negative_label]['recall']
    specificity = classification_metrics[positive_label]['recall']

    accuracy = classification_metrics['accuracy']
    conf_matrix = confusion_matrix(target_list.tolist(), pred_list.tolist())
    roc_score = roc_auc_score(target_list.tolist(), score_list.tolist())

    if plot_roc_curve:
        fpr, tpr, _ = roc_curve(target_list.tolist(), score_list.tolist())
        plt.plot(fpr, tpr, label="Area under ROC = {:.4f}".format(roc_score))
        plt.legend(loc='best')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()

    if verbose:
        print('------------------- Test Performance --------------------------------------')
        print("Accuracy \t {:.3f}".format(accuracy))
        print("Sensitivity \t {:.3f}".format(sensitivity))
        print("Specificity \t {:.3f}".format(specificity))
        print("Area Under ROC \t {:.3f}".format(roc_score))
        print("------------------------------------------------------------------------------")

    return {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Roc_score": roc_score,
        "Confusion Matrix": conf_matrix,
        "Validation Loss": val_loss / len(test_loader),
        "score_list": score_list.tolist(),
        "pred_list": pred_list.tolist(),
        "target_list": target_list.tolist(),
        "paths": path_list
    }


class EarlyStopping(object):
    def __init__(self, patience=8):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.previous_loss = int(1e8)
        self.previous_accuracy = 0
        self.init = False
        self.accuracy_decrease_iters = 0
        self.loss_increase_iters = 0
        self.best_running_accuracy = 0
        self.best_running_loss = int(1e7)

    def __call__(self, model, loss, accuracy):
        is_best = False

        # compute moving average
        if not self.init:
            running_loss = loss
            running_accuracy = accuracy
            self.init = True

        else:
            running_loss = 0.2 * loss + 0.8 * self.previous_loss
            running_accuracy = 0.2 * accuracy + 0.8 * self.previous_accuracy

        # check if running accuracy has improved beyond the best running accuracy recorded so far
        if running_accuracy < self.best_running_accuracy:
            self.accuracy_decrease_iters += 1
        else:
            self.best_running_accuracy = running_accuracy
            self.accuracy_decrease_iters = 0

        # check if the running loss has decreased from the best running loss recorded so far
        if running_loss > self.best_running_loss:
            self.loss_increase_iters += 1
        else:
            self.best_running_loss = running_loss
            self.loss_increase_iters = 0
            is_best = True

        # log the current accuracy and loss
        self.previous_accuracy = running_accuracy
        self.previous_loss = running_loss

        return is_best

    def stop(self):

        # compute thresholds
        accuracy_threshold = self.accuracy_decrease_iters > self.patience
        loss_threshold = self.loss_increase_iters > self.patience

        # return codes corresponding to exhuaustion of patience for either accuracy or loss
        # or both of them
        if accuracy_threshold and loss_threshold:
            return 1

        if accuracy_threshold:
            return 2

        if loss_threshold:
            return 3

        return 0

    def reset(self):
        # reset
        self.accuracy_decrease_iters = 0
        self.loss_increase_iters = 0


def train(model, optimizer, train_loader, val_loader, log_dir, early_stopper, learning_rate, save_path, device=None, epochs=60):
    best_val_score = 0
    writer = SummaryWriter(log_dir)
    if device is None:
        device = get_device()

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0

        for iter_num, data in enumerate(train_loader):
            image, target = data['img'].to(device), data['label'].to(device)

            # Compute the loss
            output = model(image)
            loss = criterion(output, target.long()) / 8

            # Log loss
            train_loss += loss.item()
            loss.backward()

            # Perform gradient udpate
            if iter_num % 8 == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Calculate the number of correctly classified examples
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.long().view_as(pred)).sum().item()

        # Compute and print the performance metrics
        metrics_dict = compute_metrics(model, val_loader)
        print('------------------ Epoch {} Iteration {}--------------------------------------'.format(epoch,
                                                                                                      iter_num))
        print("Accuracy \t {:.3f}".format(metrics_dict['Accuracy']))
        print("Sensitivity \t {:.3f}".format(metrics_dict['Sensitivity']))
        print("Specificity \t {:.3f}".format(metrics_dict['Specificity']))
        print("Area Under ROC \t {:.3f}".format(metrics_dict['Roc_score']))
        print("Val Loss \t {}".format(metrics_dict["Validation Loss"]))
        print("------------------------------------------------------------------------------")

        # Save the model with best validation accuracy
        if metrics_dict['Accuracy'] > best_val_score:
            torch.save(model, "best_model.pkl")
            best_val_score = metrics_dict['Accuracy']

        # print the metrics for training data for the epoch
        print('\nTraining Performance Epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
                   100.0 * train_correct / len(train_loader.dataset)))

        # log the accuracy and losses in tensorboard
        writer.add_scalars("Losses", {'Train loss': train_loss / len(train_loader),
                                      'Validation_loss': metrics_dict["Validation Loss"]},
                           epoch)
        writer.add_scalars("Accuracies", {"Train Accuracy": 100.0 * train_correct / len(train_loader.dataset),
                                          "Valid Accuracy": 100.0 * metrics_dict["Accuracy"]}, epoch)

        # Add data to the EarlyStopper object
        if early_stopper(model, metrics_dict['Validation Loss'], metrics_dict['Accuracy']):
            print("Validation loss decreased saving model ...")
            torch.save(model.state_dict(), save_path)

        # If both accuracy and loss are not improving, stop the training
        if early_stopper.stop() == 1:
            break

        # if only loss is not improving, lower the learning rate
        if early_stopper.stop() == 3:
            for param_group in optimizer.param_groups:
                learning_rate *= 0.1
                param_group['lr'] = learning_rate
                print('Updating the learning rate to {}'.format(learning_rate))
                early_stopper.reset()

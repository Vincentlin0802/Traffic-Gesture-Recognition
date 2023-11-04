import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import os

class RNNTrainer:
    def __init__(self, model, device, train_loader, test_loader, optimizer, save_model_path,total_epochs):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.save_model_path = save_model_path
        self.total_epochs = total_epochs

    def train(self, epoch):
        self.model.train()
        losses = []
        scores = []
        N_count = 0

        for batch_idx, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device).view(-1, )
            N_count += X.size(0)
            self.optimizer.zero_grad()

            output = self.model(X)
            loss = F.cross_entropy(output, y)
            losses.append(loss.item())
            y_pred = torch.max(output, 1)[1]
            step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
            scores.append(step_score)

            loss.backward()
            self.optimizer.step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(self.train_loader.dataset), 
                100. * (batch_idx + 1) / len(self.train_loader), loss.item(), 100 * step_score))

        return losses, scores

    def test(self, epoch, best_test_loss, best_epoch):
        self.model.eval()
        test_loss = 0
        all_y = []
        all_y_pred = []

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device).view(-1, )

                output = self.model(X)
                loss = F.cross_entropy(output, y, reduction='sum')
                test_loss += loss.item()

                y_pred = output.max(1, keepdim=True)[1]
                all_y.extend(y)
                all_y_pred.extend(y_pred)

        test_loss /= len(self.test_loader.dataset)
        all_y = torch.stack(all_y, dim=0)
        all_y_pred = torch.stack(all_y_pred, dim=0)
        test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

        print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            len(all_y), test_loss, 100* test_score))

        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

        #save the best test loss, and the corresponding best epoch
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch

        #save the model for the best performance
        if epoch + 1 == self.total_epochs:
            torch.save(self.model.state_dict(), os.path.join(self.save_model_path, 'rnn_epoch{}.pth'.format(best_epoch + 1)))   
            
        
        return test_loss, test_score , best_test_loss, best_epoch, all_y, all_y_pred
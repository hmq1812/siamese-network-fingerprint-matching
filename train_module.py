import torch
import numpy as np
from tqdm import tqdm

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, start_epoch=0):
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()
        print('Epoch: {}/{}'.format(epoch + 1, n_epochs))

        # Train stage
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, cuda)

        message = '\nAverage training loss: {:.4f}. '.format(train_loss)

        val_loss = test_epoch(val_loader, model, loss_fn, cuda)
        val_loss /= len(val_loader)

        message += 'Average validating loss: {:.4f}'.format(val_loss)
        print(message)

        torch.save(model.state_dict(), "model_{}.pth".format(epoch), _use_new_zipfile_serialization=False)
        print('Saving model...')

        print('\n' + '='*80 + '\n')

def train_epoch(train_loader, model, loss_fn, optimizer, cuda):
    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training epoch", position=0, leave=False)):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        losses = []

    total_loss /= (batch_idx + 1)
    return total_loss


def test_epoch(val_loader, model, loss_fn, cuda):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(val_loader, desc="Validating epoch", position=0, leave=False)):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

    return val_loss


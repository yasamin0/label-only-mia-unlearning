import logging
from pyexpat import model
import torch
from torch import nn
from runx.logx import logx
from modules import CNN
from utils import init_func
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(args, model, train_loader, model_type, save_path=None):
    logging.info(f'training {model_type} model in {args.dataset_name}')

    import random
    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = model.to(device)
    model.train()

    loss_func = nn.CrossEntropyLoss()
    correct = 0

    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[int(args.epochs*0.5), int(args.epochs*0.75)],
    gamma=0.1
)
    
    for epoch in range(1, args.epochs + 1):
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            output = model(x)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
            acc = correct / total
            logging.info(f"[{model_type}] epoch={epoch} train-acc={acc:.4f}")
            model.train()



    if save_path:
        torch.save(model, save_path)
        logging.info(f"model saved to {save_path}")



# def train_target_module(args, module, train_loader, epoch, optimizer):
#     module.train()
#     loss_func = nn.CrossEntropyLoss()
#     for step, (x, y) in enumerate(train_loader):
#         x = x.cuda()
#         y = y.cuda()
#         output = module(x)
#         loss = loss_func(output, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if step % args.log_interval == 0:
#             logx.msg('TargetModel Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch,
#                 step * len(train_loader),
#                 len(train_loader.dataset),
#                 100. * step / len(train_loader),
#                 loss.item()))
#
#
# def train_shadow_module(args, target_model, shadow_module, shadow_dataset_loader, epoch, optimizer):
#     target_model.train()
#     shadow_module.train()
#     dataset = args.datasets[args.dataset_ID]
#     # state_dict, _ = logx.load_model(path=args.logdir+dataset+'/'+str(3000)+'/target'+'/best_checkpoint_ep.pth')
#     # target_model = CNN('CNN7', dataset)
#     # target_model.load_state_dict(state_dict)
#     # target_model = torch.load(args.logdir+dataset+'/'+str(3000)+'/target_module.pt')
#     loss_func = nn.CrossEntropyLoss()
#     for step, (x, _) in enumerate(shadow_dataset_loader):
#         x = x.cuda()
#         _, y = target_model(x).max(1)
#         optimizer.zero_grad()
#         output = shadow_module(x)
#         loss = loss_func(output, y)
#         loss.backward()
#         optimizer.step()
#         if step % args.log_interval == 0:
#             logx.msg('ShadowModel Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch,
#                 step * len(x),
#                 len(shadow_dataset_loader.dataset),
#                 100. * step / len(shadow_dataset_loader),
#                 loss.item()))
#
#
# def save_shadow_module(args, module, epoch):
#     save_dict = {
#         'epoch': epoch + 1,
#         'state_dict': module.state_dict(),
#     }
#     logx.save_model(
#         save_dict,
#         epoch='',
#         higher_better=True
#     )
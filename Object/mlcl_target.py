import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import time

def cosine_similarity(feature, pairs):

    similarity = feature.mm(pairs.t())
    return similarity

def split_target(args):
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    txt_tar = open(args.t_dset_path).readlines()
    dset_loaders = {}

    test_set = ImageList(txt_tar, transform=test_transform)
    dset_loaders["target"] = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size*3,
        shuffle=False, num_workers=args.worker, drop_last=False)

    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    if args.model == "source":
        modelpath = args.output_dir_src + '/source_F.pt'
        netF.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir_src + '/source_B.pt'
        netB.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir_src + '/source_C.pt'
        netC.load_state_dict(torch.load(modelpath))
    else:
        modelpath = args.output_dir + "/target_F_" + args.savename + ".pt"
        netF.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir + "/target_B_" + args.savename + ".pt"
        netB.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir + "/target_C_" + args.savename + ".pt"
        netC.load_state_dict(torch.load(modelpath))

    netF.eval()
    netB.eval()
    netC.eval()

    start_test = True
    with torch.no_grad():
        iter_test = iter(dset_loaders['target'])
        for i in range(len(dset_loaders['target'])):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    top_pred, predict = torch.max(all_output, 1)
    acc = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0]) * 100
    mean_ent = loss.Entropy(nn.Softmax(dim=1)(all_output))

    if args.dset == 'VISDA-C':
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int),:]
        all_acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        acc = all_acc.mean()
        aa = [str(np.round(i, 2)) for i in all_acc]
        acc_list = ' '.join(aa)
        print(acc_list)
        args.out_file.write(acc_list + '\n')
        args.out_file.flush()

    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%; Mean Ent = {:.4f}'.format(args.name, 0, 0, acc, mean_ent.mean())
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    if args.ps == 0:
        est_p = (mean_ent < mean_ent.mean()).sum().item() / mean_ent.size(0)
        log_str = 'est_p: {:.2f}'.format(est_p)
        print(log_str + '\n')
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        PS = est_p
    else:
        PS = args.ps

    print("PS", PS)

    if args.choice == "ent":
        value = mean_ent
    elif args.choice == "maxp":
        value = - top_pred
    elif args.choice == "marginp":
        pred, _ = torch.sort(all_output, 1)
        value = pred[:, 1] - pred[:, 0]
    else:
        value = torch.rand(len(mean_ent))

    ori_target = txt_tar.copy()
    new_tar = []
    new_src = []

    predict = predict.numpy()

    cls_k = args.class_num
    for c in range(cls_k):
        c_idx = np.where(predict == c)
        c_idx = c_idx[0]
        c_value = value[c_idx]

        _, idx_ = torch.sort(c_value)
        c_num = len(idx_)
        c_num_s = int(c_num * args.psva)

        for ei in range(0, c_num_s):
            ee = c_idx[idx_[ei]]
            #ee是在原来的list 中的索引
            reci = ori_target[ee].strip().split(' ')
            line = reci[0] + ' ' + reci[1] + '\n'
            new_src.append(line)
        for ei in range(c_num_s, c_num):
            ee = c_idx[idx_[ei]]
            reci = ori_target[ee].strip().split(' ')
            line = reci[0] + ' ' + reci[1] + '\n'
            new_tar.append(line)


    pseudo_src = ImageList(new_src, transform=test_transform)
    pseudo_src_loader = torch.utils.data.DataLoader(pseudo_src, batch_size=args.batch_size*3,
        shuffle=False, num_workers=args.worker, drop_last=False)

    start_test = True
    with torch.no_grad():
        iter_test = iter(pseudo_src_loader)
        for i in range(len(pseudo_src_loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            rho = np.random.beta(0.75, 0.75)
            ind = torch.randperm(inputs.size(0))

            input_a, input_b = inputs, inputs[ind]

            mixed_input = rho * input_a + (1 - rho) * input_b
            feas = netB(netF(mixed_input))

            feas_a = netB(netF(input_a))
            outputs_a = netC(feas_a)

            feas_b = netB(netF(input_b))
            outputs_b = netC(feas_b)

            outputs = rho * outputs_a + (1 - rho) * outputs_b
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])


    return new_src.copy(), new_tar.copy(), initc



def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def aug_transform(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

def data_load(args):
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=[image_train(), aug_transform()])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders



def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def crossdomain_loss(initc, feats_w, pseudo_labels_w):

    nll = nn.NLLLoss()
    contrastive_label = torch.tensor([0]).cuda()
    total_contrastive_loss = torch.tensor(0.).cuda()
    feats_w = torch.cat((feats_w, torch.ones(feats_w.size(0), 1).cuda()), 1)
    feats_w = (feats_w.t() / torch.norm(feats_w, p=2, dim=1)).t()
    for idx in range(feats_w.shape[0]):
        feature_pos = torch.from_numpy(initc[pseudo_labels_w[idx]]).unsqueeze(0).cuda()
        negative_pairs = torch.Tensor([]).cuda()
        for idxc in range(initc.shape[0]):
            if idxc != pseudo_labels_w[idx]:
                negative_pairs = torch.cat((negative_pairs, torch.from_numpy(initc[idxc]).unsqueeze(0).cuda()))

        pairs4q = torch.cat((feature_pos, negative_pairs))
        result = cosine_similarity(feats_w[idx].unsqueeze(0).float().cuda(), pairs4q.float())

        numerator = torch.exp((result[0][0]) / 0.07)
        denominator = numerator + torch.sum(torch.exp((result / 0.07)[0][1:]))

        result = torch.log(numerator / denominator).unsqueeze(0).unsqueeze(0)
        contrastive_loss = nll(result, contrastive_label)
        total_contrastive_loss = total_contrastive_loss + contrastive_loss
    total_contrastive_loss = total_contrastive_loss / (feats_w.shape[0])

    return total_contrastive_loss





def train_target(args, initc):

    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    t0 = time.time()
    while iter_num < max_iter:
        try:
            inputs_all, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_all, _, tar_idx = next(iter_test)

        inputs_test = inputs_all[0].cuda()
        inputs_aug = inputs_all[1].cuda()

        if inputs_test.size(0) == 1:
            continue
            #and args.cls_par > 0:
        if iter_num % interval_iter == 0:
            netF.eval()
            netB.eval()
            mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()

        inputs_test = inputs_test.cuda()
        inputs_aug = inputs_aug.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        features_aug = netB(netF(inputs_aug))
        outputs_aug = netC(features_aug)


        classifier_loss = torch.tensor(0.0).cuda()

        if args.aug_par > 0:
            pred = mem_label[tar_idx]
            aug_weak_loss = nn.CrossEntropyLoss()(outputs_aug, pred)
            aug_weak_loss *= args.aug_par
            if iter_num < interval_iter and args.dset == "VISDA-C":
                aug_weak_loss *= 0
            classifier_loss += aug_weak_loss

        if args.cross_par > 0:
            cross_loss = crossdomain_loss(initc, features_aug, mem_label[tar_idx])
            cross_loss *= args.cross_par
            classifier_loss += cross_loss

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            t1 = time.time()
            netF.eval()
            netB.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netB.train()
            print('run time: {:.5f}'.format(t1-t0))
            t0 = time.time()


    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>args.threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return predict.astype('int')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office_home', choices=['VISDA-C', 'office', 'office_home', 'domainnet-126'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--cross_par', type=float, default=0.1)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--aug_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--psva', type=float, default=0.4)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--ps', type=float, default=0.0)
    parser.add_argument('--choice', type=str, default="ent", choices=["maxp", "ent", "marginp", "random"])
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--model', type=str, default="source", choices=["source", 'target'])
    parser.add_argument('--issave', type=bool, default=False)
    args = parser.parse_args()

    if args.dset == 'office_home':
        names = ['Art', 'Clipart', 'Product', 'Real']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'domainnet-126':
        names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 126

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = '/home/zhangzhengjie/data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office_home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        print("path",args.output_dir_src)
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'cross_fea-aug_' + str(args.cross_par) + '_aug_' + str(args.aug_par) + '_ent_' + str(args.ent_par) + '_ps_' + str(args.psva)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()

        pseudo_src, _, initc = split_target(args)

        train_target(args, initc)

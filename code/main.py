from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset, prepare_data
from trainer import condGANTrainer as trainer
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET
from model import RNN_ENCODER, CNN_ENCODER
from miscc.utils import mkdir_p
from PIL import Image
from torch.autograd import Variable
import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import socket
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
output_dir = '/output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_DMGAN.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--NET_G', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def train(gpu, args):
    start_t = time.time()
    rank = args.nr * args.gpu_id + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    # torch.cuda.set_device(args.GPU_ID)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(gpu)

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    dataset = TextDataset(cfg.DATA_DIR, "train",
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
   # assert dataset
   # subset
   # dataset_size = dataset.__len__()
   # indices = list(range(dataset_size))
   # split = int(np.floor(0.95 * dataset_size))

   # train_indices = indices[split:]
   # sub_dataset = torch.utils.data.Subset(dataset, train_indices)


    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=cfg.TRAIN.BATCH_SIZE,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)

    algo = trainer(output_dir, train_loader, dataset.n_words, dataset.ixtoword, dataset)

    text_encoder, image_encoder, netG, netsD, start_epoch = algo.build_models(gpu)
    avg_param_G = copy_G_params(netG)
    optimizerG, optimizersD = algo.define_optimizers(netG, netsD)
    real_labels, fake_labels, match_labels = algo.prepare_labels()

    batch_size = algo.batch_size
    nz = cfg.GAN.Z_DIM
    noise = Variable(torch.FloatTensor(batch_size, nz))
    fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
    if cfg.CUDA:
        noise, fixed_noise = noise.cuda(non_blocking=True), fixed_noise.cuda(non_blocking=True)

    gen_iterations = 0

    for epoch in range(start_epoch, algo.max_epoch):
        start_t = time.time()

        data_iter = iter(algo.data_loader)
        step = 0
        while step < algo.num_batches:
            # reset requires_grad to be trainable for all Ds
            # self.set_requires_grad_value(netsD, True)

            ######################################################
            # (1) Prepare training data and Compute text embeddings
            ######################################################
            data = data_iter.next()
            imgs, captions, cap_lens, class_ids, keys, attrs = prepare_data(data)
            
            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            # attrs processing
            
            attr_len = torch.Tensor([cfg.MAX_ATTR_LEN] * cap_lens.size(0))
            _, attr_emb0 = text_encoder(attrs[:, 0:1, :].squeeze(), attr_len, hidden)
            _, attr_emb1 = text_encoder(attrs[:, 1:2, :].squeeze(), attr_len, hidden)
            _, attr_emb2 = text_encoder(attrs[:, 2:3, :].squeeze(), attr_len, hidden)
            attr_embs = torch.stack((attr_emb0, attr_emb1, attr_emb2), dim=2)
            attr_embs = attr_embs.detach()  # [batch_size, nef, 2 or attr_num]

            mask = (captions == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]

            #######################################################
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, attr_embs, mask, cap_lens)

            #######################################################
            # (3) Update D network
            ######################################################
            errD_total = 0
            D_logs = ''
            for i in range(len(netsD)):
                netsD[i].zero_grad()
                #  print("netsD:", i, imgs[i].shape)
                errD, log = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                               sent_emb, real_labels, fake_labels, gpu)
                # backward and update parameters
                errD.backward()
                optimizersD[i].step()
                errD_total += errD
                D_logs += 'errD%d: %.2f ' % (i, errD.item())
                D_logs += log

            #######################################################
            # (4) Update G network: maximize log(D(G(z)))
            ######################################################
            # compute total loss for training G
            step += 1
            gen_iterations += 1

            # do not need to compute gradient for Ds
            # self.set_requires_grad_value(netsD, False)
            netG.zero_grad()
            errG_total, G_logs = \
                generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                               words_embs, sent_emb, attr_embs, match_labels, cap_lens, class_ids, gpu)
            kl_loss = KL_loss(mu, logvar)
            errG_total += kl_loss
            G_logs += 'kl_loss: %.2f ' % kl_loss.item()
            # backward and update parameters
            errG_total.backward()
            optimizerG.step()
            for p, avg_p in zip(netG.parameters(), avg_param_G):
                avg_p.mul_(0.999).add_(0.001, p.data)

            # save images
#            if gen_iterations % 10000 == 0:
#                backup_para = copy_G_params(netG)
#                load_params(netG, avg_param_G)

#                for j in range(batch_size):
#                    k = -1
#                    im0 = fake_imgs[0][j].data.cpu().numpy()
#                    im1 = fake_imgs[1][j].data.cpu().numpy()
#                    im2 = fake_imgs[2][j].data.cpu().numpy()
#                    im3 = fake_imgs[3][j].data.cpu().numpy()
                    # [-1, 1] --> [0, 255]
#                    im0 = (im0 + 1.0) * 127.5
#                    im1 = (im1 + 1.0) * 127.5
#                    im2 = (im2 + 1.0) * 127.5
#                    im3 = (im3 + 1.0) * 127.5
#                    im0 = im0.astype(np.uint8)
#                    im1 = im1.astype(np.uint8)
#                    im2 = im2.astype(np.uint8)
#                    im3 = im3.astype(np.uint8)
#                    im0 = np.transpose(im0, (1, 2, 0))
#                    im1 = np.transpose(im1, (1, 2, 0))
#                    im2 = np.transpose(im2, (1, 2, 0))
#                    im3 = np.transpose(im3, (1, 2, 0))
#                    im0_ = np.array(Image.fromarray(im0).resize((256, 256)))
#                    im1_ = np.array(Image.fromarray(im1).resize((256, 256)))
#                    im2_ = np.array(Image.fromarray(im2).resize((256, 256)))
#                    im3_ = np.array(Image.fromarray(im3).resize((256, 256)))
#                    im = np.hstack((im0_, im1_, im2_, im3_))
#                    im = Image.fromarray(im)
#                    save_path = "/apdcephfs/share_1290939/arsalruan/bird_0208/"
#                    filename = keys[j].replace("/", "_")
#                    fullpath = '%s%d_%d_%s.png' % (save_path, epoch, step, filename)
#                    im.save(fullpath)

#                load_params(netG, backup_para)

        end_t = time.time()
        if gpu == 0:
           print('''[%d/%d] Loss_D: %.2f Loss_G: %.2f Time: %.2fs''' % (
            epoch, algo.max_epoch, errD_total.item(), errG_total.item(), end_t - start_t))
    
    if gpu == 0:
        algo.save_model(netG, avg_param_G, netsD, algo.max_epoch)
   # end_t = time.time()
   # print('Total time for training:', end_t - start_t)


def sampling(gpu, args):
    start_t = time.time()
 #   rank = args.nr * args.gpu_id + gpu
 #   dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    torch.cuda.set_device(gpu)
    if cfg.TRAIN.NET_G == '':
        print('Error: the path for morels is not found!')
    else:
        # Build and load the generator
        if cfg.GAN.B_DCGAN:
            netG = G_DCGAN()
        else:
            netG = G_NET()
        netG.apply(weights_init)
        netG.cuda(gpu)
        netG.eval()

        # Get data loader
        imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
        image_transform = transforms.Compose([
            transforms.Scale(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])

        dataset = TextDataset(cfg.DATA_DIR, "test",
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        print('dataset', dataset.n_words)
        assert dataset

  #      train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
  #                                                                      num_replicas=args.world_size,
  #                                                                      rank=rank)
        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=cfg.TRAIN.BATCH_SIZE,
                                                   drop_last=True,
                                                   shuffle=False,
                                                   num_workers=int(cfg.WORKERS))

        algo = trainer(output_dir, train_loader, dataset.n_words, dataset.ixtoword, dataset)

        # load text encoder
        # print(algo.n_words)
        text_encoder = RNN_ENCODER(algo.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        # print('state_dict:', state_dict.keys(), state_dict['encoder.weight'].shape)
        # print(text_encoder.encoder.weight.shape)
        text_encoder.load_state_dict(state_dict)
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder = text_encoder.cuda(gpu)
        text_encoder.eval()

        # load image encoder
        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        print('Load image encoder from:', img_encoder_path)
        image_encoder = image_encoder.cuda(gpu)
        image_encoder.eval()

        batch_size = algo.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
        noise = noise.cuda(non_blocking=True)

        model_dir = cfg.TRAIN.NET_G
        state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
        # state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict)
        print('Load G from: ', model_dir)

        # the path to save generated images
        s_tmp = model_dir[:model_dir.rfind('.pth')]
        save_dir = '%s/%s' % (s_tmp, "valid")
        mkdir_p(save_dir)

        cnt = 0
        R_count = 0
        R = np.zeros(30000)
        cont = True
        for ii in range(5):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
            if (cont == False):
                break
            for step, data in enumerate(algo.data_loader, 0):
                cnt += batch_size
                if (cont == False):
                    break
                if step % 100 == 0:
                    print('cnt: ', cnt)
                # if step > 50:
                #     break

                imgs, captions, cap_lens, class_ids, keys, attrs = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                # attrs processing
                attr_len = torch.Tensor([cfg.MAX_ATTR_LEN] * cap_lens.size(0))
                _, attr_emb0 = text_encoder(attrs[:, 0:1, :].squeeze(), attr_len, hidden)
                _, attr_emb1 = text_encoder(attrs[:, 1:2, :].squeeze(), attr_len, hidden)
                _, attr_emb2 = text_encoder(attrs[:, 2:3, :].squeeze(), attr_len, hidden)
                attr_embs = torch.stack((attr_emb0, attr_emb1, attr_emb2), dim=2)  # [batch_size, nef, attr_num]

                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, attr_embs, mask, cap_lens)
                for j in range(batch_size):
                    s_tmp = '%s/single/%s' % (save_dir, keys[j])
                    folder = s_tmp[:s_tmp.rfind('/')]
                    if not os.path.isdir(folder):
                        # print('Make a new folder: ', folder)
                        mkdir_p(folder)
                    k = -1
                    # for k in range(len(fake_imgs)):
                    im = fake_imgs[k][j].data.cpu().numpy()
                    # [-1, 1] --> [0, 255]
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    fullpath = '%s_s%d_%d.png' % (s_tmp, k, ii)
                    im.save(fullpath)

                _, cnn_code = image_encoder(fake_imgs[-1])

                for i in range(batch_size):
                    mis_captions, mis_captions_len = algo.dataset.get_mis_caption(class_ids[i])
                    hidden = text_encoder.init_hidden(99)
                    _, sent_emb_t = text_encoder(mis_captions, mis_captions_len, hidden)
                    rnn_code = torch.cat((sent_emb[i, :].unsqueeze(0), sent_emb_t), 0)
                    ### cnn_code = 1 * nef
                    ### rnn_code = 100 * nef
                    scores = torch.mm(cnn_code[i].unsqueeze(0), rnn_code.transpose(0, 1))  # 1* 100
                    cnn_code_norm = torch.norm(cnn_code[i].unsqueeze(0), 2, dim=1, keepdim=True)
                    rnn_code_norm = torch.norm(rnn_code, 2, dim=1, keepdim=True)
                    norm = torch.mm(cnn_code_norm, rnn_code_norm.transpose(0, 1))
                    scores0 = scores / norm.clamp(min=1e-8)
                    if torch.argmax(scores0) == 0:
                        R[R_count] = 1
                    R_count += 1

                if R_count >= 30000:
                    sum = np.zeros(10)
                    np.random.shuffle(R)
                    for i in range(10):
                        sum[i] = np.average(R[i * 3000:(i + 1) * 3000 - 1])
                    R_mean = np.average(sum)
                    R_std = np.std(sum)
                    print("R mean:{:.4f} std:{:.4f}".format(R_mean, R_std))
                    cont = False
    end_t = time.time()
    print('Total time for training:', end_t - start_t)


def main():
    args = parse_args()
    args.world_size = args.gpu_id * args.nodes
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    os.environ['MASTER_ADDR'] = IPAddr
    os.environ['MASTER_PORT'] = '8888'
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.NET_G != '':
        cfg.TRAIN.NET_G = args.NET_G

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    print("Seed: %d" % (args.manualSeed))

   # if cfg.TRAIN.FLAG:
   #    mp.spawn(train, nprocs=args.gpu_id, args=(args,))
   #    train(0, args)
   # else:
    sampling(0, args)


if __name__ == "__main__":
    main()




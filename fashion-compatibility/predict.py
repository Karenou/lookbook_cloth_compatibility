from __future__ import print_function
import argparse
import os
from PIL import Image
from matplotlib.cbook import file_requires_unicode
import pandas as pd
import h5py

import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader 

import Resnet_18
from tripletnet import Tripletnet
from type_specific_network import TypeSpecificNet

# Training settings
parser = argparse.ArgumentParser(description='Fashion Compatibility Example')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before logging testing status')
parser.add_argument('--dim_embed', type=int, default=64, metavar='N',
                    help='how many dimensions in embedding (default: 64)')
parser.add_argument('--use_fc', action='store_true', default=False,
                    help='Use a fully connected layer to learn type specific embeddings.')
parser.add_argument('--learned', dest='learned', action='store_true', default=False,
                    help='To learn masks from random initialization')
parser.add_argument('--prein', dest='prein', action='store_true', default=False,
                    help='To initialize masks to be disjoint')
parser.add_argument('--rand_typespaces', action='store_true', default=False,
                    help='randomly assigns comparisons to type-specific embeddings where #comparisons < #embeddings')
parser.add_argument('--num_rand_embed', type=int, default=1, metavar='N',
                    help='number of random embeddings when rand_typespaces=True')
parser.add_argument('--l2_embed', dest='l2_embed', action='store_true', default=False,
                    help='L2 normalize the output of the type specific embeddings')
parser.add_argument('--learned_metric', dest='learned_metric', action='store_true', default=False,
                    help='Learn a distance metric rather than euclidean distance')
parser.add_argument('--margin', type=float, default=0.3, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--embed_loss', type=float, default=5e-4, metavar='M',
                    help='parameter for loss for embedding norm')
parser.add_argument('--mask_loss', type=float, default=5e-4, metavar='M',
                    help='parameter for loss for mask norm')
parser.add_argument('--vse_loss', type=float, default=5e-3, metavar='M',
                    help='parameter for loss for the visual-semantic embedding')
parser.add_argument('--sim_t_loss', type=float, default=5e-5, metavar='M',
                    help='parameter for loss for text-text similarity')
parser.add_argument('--sim_i_loss', type=float, default=5e-5, metavar='M',
                    help='parameter for loss for image-image similarity')

parser.add_argument('--resume', default='', type=str,
                    help='path to load pretrain model')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--base_path', type=str,
                    help='path to load cloth segent images')
parser.add_argument('--segmentation_csv', type=str,
                    help='path to load segmentation output csv')
parser.add_argument('--output_score', type=str,
                    help='path to save compatibility score csv')
parser.add_argument('--output_mapping', type=str,
                    help='path to save mapping of embeded vector to look_id and item_id')
parser.add_argument('--output_hdf5', type=str,
                    help='path to save embedding vector as hdf5 file')


CATEGORY_MAPPING = {0: 'background', 1: 'upper body', 2: 'lower body', 3: 'full body'}


def filter_multiple_items(df):
    """
    this function filter the dataframe with rows whose number detected items is larger then 1
    """
    df = df[df["segmentation_cnt"] > 1]
    df = df.reset_index(drop=True)
    return df


class ImageDataset(Dataset):
    def __init__(self, base_path, csv_path, resume_idx=0):
        super(Dataset, self).__init__()
        self.base_path = base_path
        self.transform =  transforms.Compose([
            transforms.Resize(112),
            transforms.CenterCrop(112),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        # only read images if detected more than 1 items        
        self.mapping = pd.read_csv(csv_path)
        self.mapping = filter_multiple_items(self.mapping)

        if resume_idx > 0:
            self.mapping = self.mapping[resume_idx:].reset_index(drop=True)

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):
        user_id = self.mapping["user_id"][index]
        look_id = self.mapping["look_id"][index]

        # create a (3, 3, 112, 112) tensor to store the segments
        # if only two items are detected, leave one dimension filled with 0
        # use a marker to record which dimension is non-zero
        marker = torch.zeros(3)
        imgs = torch.zeros((3, 3, 112, 112))
        
        for idx in range(3):
            path = "%s/%d_%d_%d.jpg" % (self.base_path, user_id, look_id, idx + 1)
            if os.path.exists(path):
                marker[idx] = 1
                img = Image.open(path)
                img = self.transform(img)
                imgs[idx, :, :, :] = img
        # print(user_id, look_id, marker)
        return user_id, look_id, marker, imgs
    

def write_to_hdfs(file, embed_vec, user_id, look_id):
    if str(user_id) not in file.keys():
        group = file.create_group(str(user_id))
    else:
        group = file[str(user_id)]
    
    group[str(look_id)] = embed_vec.cpu().detach().numpy()
    return file


if __name__ == "__main__":
    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    dataset = ImageDataset(args.base_path, args.segmentation_csv)

    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    test_loader = DataLoader(
        dataset, batch_size=args.batch_size, 
        shuffle=False, drop_last=False,  **kwargs
    )

    # load model
    model = Resnet_18.resnet18(pretrained=True, embedding_size=args.dim_embed)
    # number of type spaces, set to 4
    csn_model = TypeSpecificNet(args, model, args.num_rand_embed)
    criterion = torch.nn.MarginRankingLoss(margin=args.margin)

    text_feature_dim = 6000
    tnet = Tripletnet(args, csn_model, text_feature_dim, criterion)
    tnet.eval()
    if args.cuda:
        tnet.cuda()

    # load pretrain model checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, encoding='latin1')
            args.start_epoch = checkpoint['epoch']
            tnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    print("Predict compatability scores")
    embed_idx_mapping = []
    scores = []

    embed_hdf5 = h5py.File(args.output_hdf5, 'w')

    batch_idx = 0
    for user_id, look_id, marker, imgs in test_loader:
        if batch_idx % args.log_interval == 0:
            print("Process %d / %d" % (batch_idx, len(dataset)))
        
        marker = marker[0]
        batch_idx += 1
        if args.cuda:
            imgs = imgs.cuda()

        embeddings = []
        imgs = Variable(imgs.squeeze(axis=0))
        n_items = 0
        for item_id in range(3):
            if marker[item_id] == 1:
                # embedding shape: (1, num_rand_embed + 1, 64), the last dimension in the second axis is embedded_x
                embedding = tnet.embeddingnet(imgs[item_id].unsqueeze(axis=0)).data
                # now shape [1, dim_embed]
                embedding = embedding[:, args.num_rand_embed, :]
                embeddings.append(embedding)

                embed_idx_mapping.append([user_id.item(), look_id.item(), item_id+1, n_items])
                n_items += 1

        # write the embeded vectors to hdf5 with path: 'user_id/look_id'
        embeddings = torch.concat(embeddings)
        embed_hdf5 = write_to_hdfs(embed_hdf5, embeddings, user_id.item(), look_id.item())

        # # !!!!!if there are 2 or 3 segments, how to measure the distance?
        # outfit_vec = torch.nn.functional.pairwise_distance(
        #     embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0), p=2)
        # score = torch.norm(outfit_vec)
        # scores.append([user_id, look_id, score.cpu().detach().numpy()])

       

    print("Save embedding vectors to hdf5")
    embed_hdf5.close()
    
    print("Save embedding to look_id mapping csv")
    # item_id: cloth segment category id, idx: position in hdf5 embeded vectors
    mapping_df = pd.DataFrame(embed_idx_mapping, columns=["user_id", "look_id", "item_id", "idx"])
    mapping_df.to_csv(args.output_mapping, index=False)

    # print("Save compatability scores")
    # score_df = pd.DataFrame(scores, columns=["user_id", "look_id", "compatibility_score"])
    # score_df.to_csv(args.output_score, index=False)
    # break
    



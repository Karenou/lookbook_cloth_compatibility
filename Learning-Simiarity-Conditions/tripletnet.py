import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class ConceptBranch(nn.Module):
    def __init__(self, out_dim, embedding_dim):
        super(ConceptBranch, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(embedding_dim, 32), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(32, out_dim), nn.Softmax())
                                 
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# without VSE and Sim loss
class CS_Tripletnet(nn.Module):
    def __init__(self, embeddingnet, num_concepts):
        super(CS_Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet
        self.num_concepts = num_concepts
        self.concept_branch = ConceptBranch(self.num_concepts, 64*3)

    def forward(self, x, y, z, c):
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            c: Integer indicating according to which notion of similarity images are compared"""
       
        general_x = self.embeddingnet.embeddingnet(x)
        general_y = self.embeddingnet.embeddingnet(y)
        general_z = self.embeddingnet.embeddingnet(z)

        # l2-normalize embeddings
        norm = torch.norm(general_x, p=2, dim=1) + 1e-10
        general_x = general_x / norm.unsqueeze(-1).expand_as(general_x)
        norm = torch.norm(general_y, p=2, dim=1) + 1e-10
        general_y = general_y / norm.unsqueeze(-1).expand_as(general_y)
        norm = torch.norm(general_z, p=2, dim=1) + 1e-10
        general_z = general_z / norm.unsqueeze(-1).expand_as(general_z)

        feat = torch.cat((general_x, general_y), 1)
        feat = torch.cat((feat, general_z), 1)
        weights_xy = self.concept_branch(feat)
        embedded_x = None
        embedded_y = None
        embedded_z = None
        mask_norm = None
        for idx in range(self.num_concepts):
            concept_idx = np.zeros((len(x),), dtype=int)
            concept_idx += idx
            concept_idx = torch.from_numpy(concept_idx)
            concept_idx = concept_idx.cuda()
            concept_idx = Variable(concept_idx)

            tmp_embedded_x, masknorm_norm_x, embed_norm_x = self.embeddingnet(x, concept_idx)
            tmp_embedded_y, masknorm_norm_y, embed_norm_y = self.embeddingnet(y, concept_idx)
            tmp_embedded_z, masknorm_norm_z, embed_norm_z = self.embeddingnet(z, concept_idx)
 
            if mask_norm is None:
                mask_norm = masknorm_norm_x
            else:
                mask_norm += masknorm_norm_x

            weights = weights_xy[:, idx]
            weights = weights.unsqueeze(1)
            if embedded_x is None:
                # E_i = weight * O
                embedded_x = tmp_embedded_x * weights.expand_as(tmp_embedded_x)
                embedded_y = tmp_embedded_y * weights.expand_as(tmp_embedded_y)
                embedded_z = tmp_embedded_z * weights.expand_as(tmp_embedded_z)
            else:
                embedded_x += tmp_embedded_x * weights.expand_as(tmp_embedded_x)
                embedded_y += tmp_embedded_y * weights.expand_as(tmp_embedded_y)
                embedded_z += tmp_embedded_z * weights.expand_as(tmp_embedded_z)
        
        # image embedding: embeded_x, embedded_y, embedded_z
        # used to calculate loss: mask_norm is only for x (anchor image), while embed_norm is for (x,y,z)
        mask_norm /= self.num_concepts
        embed_norm = (embed_norm_x + embed_norm_y + embed_norm_z) / 3
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2) 
        return dist_a, dist_b, mask_norm, embed_norm



# for compute VSE and Sim loss
def selective_margin_loss(pos_samples, neg_samples, margin, has_sample):
    """ pos_samples: Distance between positive pair
        neg_samples: Distance between negative pair
        margin: minimum desired margin between pos and neg samples
        has_sample: Indicates if the sample should be used to calcuate the loss
    """
    margin_diff = torch.clamp((pos_samples - neg_samples) + margin, min=0, max=1e6)
    num_sample = max(torch.sum(has_sample), 1)
    return torch.sum(margin_diff * has_sample) / num_sample

def accuracy(pos_samples, neg_samples):
    """ pos_samples: Distance between positive pair
        neg_samples: Distance between negative pair
    """
    is_cuda = pos_samples.is_cuda
    margin = 0
    pred = (pos_samples - neg_samples - margin).cpu().data
    acc = (pred > 0).sum() * 1.0 / pos_samples.size()[0]
    acc = torch.from_numpy(np.array([acc], np.float32))
    if is_cuda:
        acc = acc.cuda()

def make_fc_1d(f_in, f_out):
    return nn.Sequential(nn.Linear(f_in, f_out),
                         nn.BatchNorm1d(f_out, eps=0.001, momentum=0.01),
                         nn.ReLU(inplace=True))


class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, embedding_dim):
        super(EmbedBranch, self).__init__()
        self.fc1 = make_fc_1d(feat_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        # L2 normalize each feature vector
        norm = torch.norm(x, p=2, dim=1) + 1e-10
        x = x / norm.expand_as(x)
        return x


# with VSE and Sim loss
class Tripletnet(nn.Module):
    def __init__(self, args, embeddingnet, num_concepts, text_dim, criterion):
        super(CS_Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet
        self.num_concepts = num_concepts
        self.text_branch = EmbedBranch(text_dim, args.dim_embed)
        self.concept_branch = ConceptBranch(self.num_concepts, 64*3)
        self.criterion = criterion
        self.margin = args.margin

    def get_image_embedding(self, x):
        """
        x: a batch of images
        return the final image representation
        """
        general_x = self.embeddingnet.embeddingnet(x)
        norm = torch.norm(general_x, p=2, dim=1) + 1e-10
        general_x_n = general_x / norm.unsqueeze(-1).expand_as(general_x)

        # concat the general embedding to feed into condition weight branch
        feat = torch.cat((general_x_n, general_x_n), 1)
        feat = torch.cat((feat, general_x_n), 1)
        weights_x = self.concept_branch(feat)
        embedded_x = None

        for idx in range(self.num_concepts):
            concept_idx = np.zeros((len(x),), dtype=int)
            concept_idx += idx
            concept_idx = torch.from_numpy(concept_idx)
            concept_idx = concept_idx.cuda()
            concept_idx = Variable(concept_idx)

            tmp_embedded_x, _, _ = self.embeddingnet(x, concept_idx)

            weights = weights_x[:, idx]
            weights = weights.unsqueeze(1)
            if embedded_x is None:
                embedded_x = tmp_embedded_x * weights.expand_as(tmp_embedded_x)
            else:
                embedded_x += tmp_embedded_x * weights.expand_as(tmp_embedded_x)
        
        return embedded_x

    def image_forward(self, x, y, z):
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            in TrainData type
            c: Integer indicating according to which notion of similarity images are compared
        """
        # general embedding after resnet-18 
        general_x = self.embeddingnet.embeddingnet(x.images)
        general_y = self.embeddingnet.embeddingnet(y.images)
        general_z = self.embeddingnet.embeddingnet(z.images)

        # l2-normalize embeddings as input to conditional weight branch
        norm = torch.norm(general_x, p=2, dim=1) + 1e-10
        general_x_n = general_x / norm.unsqueeze(-1).expand_as(general_x)
        norm = torch.norm(general_y, p=2, dim=1) + 1e-10
        general_y_n = general_y / norm.unsqueeze(-1).expand_as(general_y)
        norm = torch.norm(general_z, p=2, dim=1) + 1e-10
        general_z_n = general_z / norm.unsqueeze(-1).expand_as(general_z)

        feat = torch.cat((general_x_n, general_y_n), 1)
        feat = torch.cat((feat, general_z_n), 1)
        weights_xy = self.concept_branch(feat)
        embedded_x = None
        embedded_y = None
        embedded_z = None
        mask_norm = None
        
        for idx in range(self.num_concepts):
            concept_idx = np.zeros((len(x),), dtype=int)
            concept_idx += idx
            concept_idx = torch.from_numpy(concept_idx)
            concept_idx = concept_idx.cuda()
            concept_idx = Variable(concept_idx)

            tmp_embedded_x, masknorm_norm_x, embed_norm_x = self.embeddingnet(x, concept_idx)
            tmp_embedded_y, masknorm_norm_y, embed_norm_y = self.embeddingnet(y, concept_idx)
            tmp_embedded_z, masknorm_norm_z, embed_norm_z = self.embeddingnet(z, concept_idx)
 
            if mask_norm is None:
                mask_norm = masknorm_norm_x
            else:
                mask_norm += masknorm_norm_x

            weights = weights_xy[:, idx]
            weights = weights.unsqueeze(1)
            if embedded_x is None:
                embedded_x = tmp_embedded_x * weights.expand_as(tmp_embedded_x)
                embedded_y = tmp_embedded_y * weights.expand_as(tmp_embedded_y)
                embedded_z = tmp_embedded_z * weights.expand_as(tmp_embedded_z)
            else:
                embedded_x += tmp_embedded_x * weights.expand_as(tmp_embedded_x)
                embedded_y += tmp_embedded_y * weights.expand_as(tmp_embedded_y)
                embedded_z += tmp_embedded_z * weights.expand_as(tmp_embedded_z)
        
        # final image embedding: embeded_x, embedded_y, embedded_z
        mask_norm /= self.num_concepts
        embed_norm = (embed_norm_x + embed_norm_y + embed_norm_z) / 3
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2) 
        
        target = torch.FloatTensor(dist_a.size()).fill_(1)
        if dist_a.is_cuda:
            target = target.cuda()
        target = Variable(target)

        # triplet loss
        loss_triplet = self.criterion(dist_a, dist_b, target)
        loss_embed = embed_norm / np.sqrt(len(x))
        loss_mask = mask_norm / len(x)
        
        acc = accuracy(dist_a, dist_b)

        # calculate image similarity loss on the general embedding 
        disti_p = F.pairwise_distance(general_y, general_z, 2)
        disti_n1 = F.pairwise_distance(general_y, general_x, 2)
        disti_n2 = F.pairwise_distance(general_z, general_x, 2)
        loss_sim_i1 = self.criterion(disti_p, disti_n1, target)
        loss_sim_i2 = self.criterion(disti_p, disti_n2, target)
        loss_sim_i = (loss_sim_i1 + loss_sim_i2) / 2.

        return acc, loss_triplet, loss_sim_i, loss_mask, loss_embed, general_x, general_y, general_z

    def text_forward(self, x, y, z):
        """ x: Anchor data
            y: Distant (negative) data
            z: Close (positive) data
            in TrainData type
        """
        desc_x = self.text_branch(x.text)
        desc_y = self.text_branch(y.text)
        desc_z = self.text_branch(z.text)
        distd_p = F.pairwise_distance(desc_y, desc_z, 2)
        distd_n1 = F.pairwise_distance(desc_x, desc_y, 2)
        distd_n2 = F.pairwise_distance(desc_x, desc_z, 2)
        has_text = x.has_text * y.has_text * z.has_text
        loss_sim_t1 = selective_margin_loss(distd_p, distd_n1, self.margin, has_text)
        loss_sim_t2 = selective_margin_loss(distd_p, distd_n2, self.margin, has_text)
        loss_sim_t = (loss_sim_t1 + loss_sim_t2) / 2.
        return loss_sim_t, desc_x, desc_y, desc_z

    def calc_vse_loss(self, desc_x, general_x, general_y, general_z, has_text):
        """ Both y and z are assumed to be negatives because they are not from the same 
            item as x
            desc_x: Anchor language embedding
            general_x: Anchor visual embedding
            general_y: Visual embedding from another item from input triplet
            general_z: Visual embedding from another item from input triplet
            has_text: Binary indicator of whether x had a text description
        """
        distd1_p = F.pairwise_distance(general_x, desc_x, 2)
        distd1_n1 = F.pairwise_distance(general_y, desc_x, 2)
        distd1_n2 = F.pairwise_distance(general_z, desc_x, 2)
        loss_vse_1 = selective_margin_loss(distd1_p, distd1_n1, self.margin, has_text)
        loss_vse_2 = selective_margin_loss(distd1_p, distd1_n2, self.margin, has_text)
        return (loss_vse_1 + loss_vse_2) / 2.

    def forward(self, x, y, z):
        """ x: Anchor data
            y: Distant (negative) data
            z: Close (positive) data
            in TrainData type
        """
        acc, loss_triplet, loss_sim_i, loss_mask, loss_embed, general_x, general_y, general_z = self.image_forward(x, y, z)
        loss_sim_t, desc_x, desc_y, desc_z = self.text_forward(x, y, z)
        loss_vse_x = self.calc_vse_loss(desc_x, general_x, general_y, general_z, x.has_text)
        loss_vse_y = self.calc_vse_loss(desc_y, general_y, general_x, general_z, y.has_text)
        loss_vse_z = self.calc_vse_loss(desc_z, general_z, general_x, general_y, z.has_text)
        loss_vse = (loss_vse_x + loss_vse_y + loss_vse_z) / 3.
        return acc, loss_triplet, loss_mask, loss_embed, loss_vse, loss_sim_t, loss_sim_i
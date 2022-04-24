from PIL import Image
import os
import os.path
import torch.utils.data
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import numpy as np
np.random.seed(1)
import json
import matplotlib.pyplot as plt

filenames = {'train': ['class_tripletlist_train.txt', 'closure_tripletlist_train.txt', 
                'gender_tripletlist_train.txt', 'heel_tripletlist_train.txt'],
             'val': ['class_tripletlist_val.txt', 'closure_tripletlist_val.txt', 
                'gender_tripletlist_val.txt', 'heel_tripletlist_val.txt'],
             'test': ['class_tripletlist_test.txt', 'closure_tripletlist_test.txt', 
                'gender_tripletlist_test.txt', 'heel_tripletlist_test.txt']}

def default_image_loader(path):
    return Image.open(path).convert('RGB')

# for ut-zap50k-images dataloader
class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, base_path, filenames_filename, conditions, split, n_triplets, transform=None,
                 loader=default_image_loader):
        """ filenames_filename: A text file with each line containing the path to an image e.g.,
                images/class1/sample.jpg
            triplets_file_name: A text file with each line containing three integers, 
                where integer i refers to the i-th image in the filenames file. 
                For a line of intergers 'a b c', a triplet is defined such that image a is more 
                similar to image c than it is to image b, e.g., 
                0 2017 42 """
        self.root = root
        self.base_path = base_path  
        self.filenamelist = []
        for line in open(os.path.join(self.root, filenames_filename)):
            self.filenamelist.append(line.rstrip('\n'))
        triplets = []
        if split == 'train':
            fnames = filenames['train']
        elif split == 'val':
            fnames = filenames['val']
        else:
            fnames = filenames['test']
        #if split == 'test':
            #print(fnames)
        for condition in conditions:
            for line in open(os.path.join(self.root, 'tripletlists', fnames[condition])):
                triplets.append((line.split()[0], line.split()[1], line.split()[2], condition)) # anchor, far, close   
        # print(triplets[:100])   
        np.random.shuffle(triplets)
        # print(triplets[:100])  
        self.triplets = triplets[:int(n_triplets * 1.0 * len(conditions) / 4)]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path1, path2, path3, c = self.triplets[index]
        if os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])) and os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])) and os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])):
            img1 = self.loader(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)]))
            img2 = self.loader(os.path.join(self.root, self.base_path, self.filenamelist[int(path2)]))
            img3 = self.loader(os.path.join(self.root, self.base_path, self.filenamelist[int(path3)]))
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
            return img1, img2, img3, c
        else:
            return None

    def __len__(self):
        return len(self.triplets)



# for polyvore outfit dataloader
def parse_iminfo(question, im2index, id2im, gt=None):
    """ Maps the questions from the FITB and compatibility tasks back to
        their index in the precomputed matrix of features

        question: List of images to measure compatibility between
        im2index: Dictionary mapping an image name to its location in a
                  precomputed matrix of features
        gt: optional, the ground truth outfit set this item belongs to
    """
    questions = []
    is_correct = np.zeros(len(question), np.bool)
    for index, im_id in enumerate(question):
        set_id = im_id.split('_')[0]
        if gt is None:
            gt = set_id

        im = id2im[im_id]
        questions.append((im2index[im], im))
        is_correct[index] = set_id == gt

    return questions, is_correct, gt


def load_compatibility_questions(fn, im2index, id2im):
    """ Returns the list of compatibility questions for the
        split """
    with open(fn, 'r') as f:
        lines = f.readlines()

    compatibility_questions = []
    for line in lines:
        data = line.strip().split()
        compat_question, _, _ = parse_iminfo(data[1:], im2index, id2im)
        compatibility_questions.append((compat_question, int(data[0])))

    return compatibility_questions


def load_fitb_questions(fn, im2index, id2im):
    """ Returns the list of fill in the blank questions for the
        split """
    data = json.load(open(fn, 'r'))
    questions = []
    for item in data:
        question = item['question']
        q_index, _, gt = parse_iminfo(question, im2index, id2im)
        answer = item['answers']
        a_index, is_correct, _ = parse_iminfo(answer, im2index, id2im, gt)
        questions.append((q_index, a_index, is_correct))

    return questions


class TripletPolyvoreImageLoader(torch.utils.data.Dataset):
    def __init__(self, datadir, polyvore_split, split, conditions, 
                meta_data, triplet_path, num_triplets=None, transform=None,
                text_dim=6000, loader=default_image_loader):
        """ 
        @param datadir: base path of data folder
        @param polyvore_split: split of polyvore dataset
        @param split: train, valid or test
        @param conditions: a list of condition List[int]
        @param meta_data:
        @param triplet_path: path to load and save triplet
        @param transform
        @param text_dim: text feature dimension
        """
        self.impath = os.path.join(datadir, 'polyvore_outfits', 'images')
        self.is_train = split == 'train'
        self.loader = loader
        self.transform = transform
        self.triplet_path = "%s/%s_%s.txt" % (triplet_path, polyvore_split, split)
        self.rootdir = os.path.join(datadir, 'polyvore_outfits', polyvore_split)
        self.text_feat_dim = text_dim
        self.split = split
        
        print("Load outfit data")
        outfit_data = json.load(open(os.path.join(self.rootdir, '%s.json' % split), 'r'))   
        
        if self.is_train:
            self.im2type, self.category2ims, self.imnames = self.load_train_dict(outfit_data, meta_data)
            # huge IO time in loading text feature
            self.desc2vecs, self.im2desc = self.load_text_feature(meta_data)

            # load tripelet
            if os.path.exists(self.triplet_path):
                self.triplets = self.load_triplet(num_triplets)
            else:
                # prepare triplet
                triplets = []
                i = 0
                for outfit in outfit_data:
                    if i % 1000 == 0:
                        print("  Number of outfits: %d" % i)
                    
                    if num_triplets is not None and len(triplets) > num_triplets:
                        break

                    items = outfit['items']
                    cnt = len(items)
                    for j in range(cnt - 1):
                        for k in range(j + 1, cnt):
                            anchor_im = items[j]['item_id']
                            pos_im = items[k]['item_id']
                            item_type =  self.im2type[pos_im]
                            neg_im = self.sample_negative(pos_im, item_type)

                            # randomly assign condition
                            condition = np.random.random_integers(low=conditions[0], high=conditions[-1], size=1)[0]
                            
                            # anchor, far, close 
                            triplets.append([anchor_im, neg_im, pos_im, condition])

                    i += 1
                
                self.save_triplet(triplets)
                self.triplets = triplets

            print("%s set data size: %d"  % (split, len(self.triplets)))
        else:
            self.imnames, self.im2index, self.id2im = self.load_test_dict(outfit_data)
            # pull the two task's questions for test and val splits
            fn = os.path.join(self.rootdir, 'fill_in_blank_%s.json' % split)
            self.fitb_questions = load_fitb_questions(fn, self.im2index, self.id2im)
            fn = os.path.join(self.rootdir, 'compatibility_%s.txt' % split)
            self.compatibility_questions = load_compatibility_questions(fn, self.im2index, self.id2im)
            print("%s set data size: %d" % (split, len(self.imnames)))

    def load_train_dict(self, outfit_data, meta_data):
        print("Load dict")
        im2type = {}
        category2ims = {}
        imnames = set()
        for outfit in outfit_data:
            outfit_id = outfit['set_id']
            for item in outfit['items']:
                im = item['item_id']
                category = meta_data[im]['semantic_category']
                im2type[im] = category

                if category not in category2ims.keys():
                    category2ims[category] = {}

                if outfit_id not in category2ims[category].keys():
                    category2ims[category][outfit_id] = []

                category2ims[category][outfit_id].append(im)
                imnames.add(im)

        imnames = list(imnames)

        return im2type, category2ims, imnames

    def load_test_dict(self, outfit_data):
        print("Load dict")
        imnames = set()
        id2im = {}
        for outfit in outfit_data:
            outfit_id = outfit['set_id']
            for item in outfit['items']:
                im = item['item_id']
                id2im['%s_%i' % (outfit_id, item['index'])] = im
                imnames.add(im)

        imnames = list(imnames)

        im2index = {}
        for index, im in enumerate(imnames):
            im2index[im] = index

        return imnames, im2index, id2im       

    def load_text_feature(self, meta_data):
        """
        load text feature and description
        return desc2vecs, im2desc
        """
        print("Load text features")
        desc2vecs = {}
        featfile = os.path.join(self.rootdir, 'train_hglmm_pca6000.txt')
        with open(featfile, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                vec = line.split(',')
                label = ','.join(vec[:-self.text_feat_dim])
                vec = np.array([float(x) for x in vec[-self.text_feat_dim:]], np.float32)
                assert (len(vec) == self.text_feat_dim)
                desc2vecs[label] = vec

        im2desc = {}
        for im in self.imnames:
            desc = meta_data[im]['title']
            if not desc:
                desc = meta_data[im]['url_name']

            desc = desc.replace('\n', '').encode('ascii', 'ignore').strip().lower()

            # sometimes descriptions didn't map to any known words so they were
            # removed, so only add those which have a valid feature representation
            if desc and desc in desc2vecs:
                im2desc[im] = desc

        print("%d items have text features" % len(desc2vecs.keys()))
        return desc2vecs, im2desc

    def load_triplet(self, num_triplets):
        """
        load triplet to txt, use subset of triplets
        """
        print("load triplets from %s" % self.triplet_path)
        triplets = []
        with open(self.triplet_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                triplets.append((line.split()[0], line.split()[1], line.split()[2], line.split()[3]))

        np.random.shuffle(triplets)
        if num_triplets is not None:
            triplets = triplets[:num_triplets]
        
        return triplets

    def save_triplet(self, triplets):
        """
        save triplets to txt
        """
        print("save triplets to %s" % self.triplet_path)
        with open(self.triplet_path, "w") as f:
            for row in triplets:
                line = " ".join(map(str, row))
                f.write(line + "\n")
            f.close()

    def load_train_item(self, image_id):
        """ 
        Returns a single item in the triplet and its data
        """
        imfn = os.path.join(self.impath, '%s.jpg' % image_id)
        img = self.loader(imfn)
        if self.transform is not None:
            img = self.transform(img)

        # load text description
        if image_id in self.im2desc:
            text = self.im2desc[image_id]
            text_features = self.desc2vecs[text]
            has_text = 1
        else:
            text_features = np.zeros(self.text_feat_dim, np.float32)
            has_text = 0.

        has_text = np.float32(has_text)
        return img, text_features, has_text

    def sample_negative(self, item_id, item_type):
        """ Returns a randomly sampled item from a different set
            than the outfit at data_index, but of the same type as
            item_type

            data_index: index in self.data where the positive pair
                        of items was pulled from
            item_type: the coarse type of the item that the item
                       that was paired with the anchor
        """
        item_out = item_id
        # candidate outfits
        candidate_sets = list(self.category2ims[item_type].keys())
        attempts = 0
        while item_out == item_id and attempts < 100:
            choice = np.random.choice(candidate_sets, size=1)[0]
            items = self.category2ims[item_type][choice]
            item_index = np.random.choice(range(len(items)))
            item_out = items[item_index]
            attempts += 1
        
        return item_out

    def __getitem__(self, index):
        if self.is_train:
            # load anchor, neg, pos image and text feature if possible
            anchor_im, neg_im, pos_im, condition = self.triplets[index]
            img1, desc1, has_text1 = self.load_train_item(anchor_im)
            img2, desc2, has_text2 = self.load_train_item(neg_im)
            img3, desc3, has_text3 = self.load_train_item(pos_im)

            return img1, desc1, has_text1, img2, desc2, has_text2, img3, desc3, has_text3, condition
        else:
            # only load image
            anchor = self.imnames[index]
            img1 = self.loader(os.path.join(self.impath, '%s.jpg' % anchor))
            if self.transform is not None:
                img1 = self.transform(img1)

            return img1

    def __len__(self):
        if self.is_train:
            return len(self.triplets)

        return len(self.imnames)

    def test_compatibility(self, embeds, plot=False, save_path=None, split=None):
        """ Returns the area under a roc curve for the compatibility
            task

            embeds: precomputed embedding features used to score
                    each compatibility question
            plot_roc: whether to plot the roc curve
            save_path: path to save the roc plot
            split: which split of polyvore outfit, disjoint or nondisjoint
        """
        scores = []
        labels = np.zeros(len(self.compatibility_questions), np.int32)
        for index, (outfit, label) in enumerate(self.compatibility_questions):
            labels[index] = label
            n_items = len(outfit)
            outfit_score = 0.0
            num_comparisons = 0.0
            for i in range(n_items - 1):
                item1, _ = outfit[i]
                for j in range(i + 1, n_items):
                    item2, _ = outfit[j]
                    embed1 = embeds[item1].unsqueeze(0)
                    embed2 = embeds[item2].unsqueeze(0)
                    outfit_score += torch.nn.functional.pairwise_distance(embed1, embed2, 2)
                    num_comparisons += 1.

            outfit_score /= num_comparisons
            scores.append(outfit_score)

        # larger scores, less compatibility, thus use 1 - scores as predicted prob
        scores = torch.cat(scores).squeeze().cpu().numpy()
        auc = roc_auc_score(labels, 1 - scores)

        if plot:
            self.plot_roc_curve(labels, 1 - scores, auc, save_path, split)
            self.plot_pr_curve(labels, 1 - scores, save_path, split)

        return auc

    def plot_roc_curve(self, label, score, auc, save_path, split):
        """
        plot roc curve
        """
        fpr, tpr, _ = roc_curve(label, score)
        plt.figure(figsize=(8,8))
        plt.plot(fpr, tpr, color = "darkorange", lw=2, label="ROC curve (auc = %.2f)" % auc)
        plt.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")
        plt.xlim([0, 1])
        plt.ylim([0,1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(
            "ROC Curve of Pretrained Model's Compatibility \n Prediction on %s Polyvore Outfit Dataset" % split
        )
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_path, "polyvore_outfit_%s_roc.eps" % split), format="eps")
        plt.close()

    def plot_pr_curve(self, label, score, save_path, split):
        """
        plot precision-recall curve
        """
        precision, recall, _ = precision_recall_curve(label, score)
        plt.figure(figsize=(8,8))
        plt.plot(recall, precision, color = "darkorange", lw=2, label="Precision-Recall curve")
        plt.xlim([0, 1])
        plt.ylim([0,1])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(
            "PR Curve of Pretrained Model's Compatibility \n Prediction on %s Polyvore Outfit Dataset" % split
        )
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_path, "polyvore_outfit_%s_pr.eps" % split), format="eps")
        plt.close()

    def test_fitb(self, embeds):
        """ Returns the accuracy of the fill in the blank task

            embeds: precomputed embedding features used to score
                    each compatibility question
        """
        correct = 0.
        n_questions = 0.
        for _, (questions, answers, is_correct) in enumerate(self.fitb_questions):
            answer_score = np.zeros(len(answers), dtype=np.float32)
            for index, (answer, _) in enumerate(answers):
                score = 0.0
                for question, _ in questions:
                    embed1 = embeds[question].unsqueeze(0)
                    embed2 = embeds[answer].unsqueeze(0)
                    score += torch.nn.functional.pairwise_distance(embed1, embed2, 2)

                answer_score[index] = score.squeeze().cpu().numpy()

            correct += is_correct[np.argmin(answer_score)]
            n_questions += 1

        # scores are based on distances so need to convert them so higher is better
        acc = correct / n_questions
        return acc

    def shuffle(self):
        np.random.shuffle(self.triplets)
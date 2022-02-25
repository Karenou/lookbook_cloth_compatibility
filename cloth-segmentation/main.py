"""
need to save the segmentation count and category for each item/cloth detected in the image
"""
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import glob
import argparse

import torch
from torch.utils.data import Dataset, DataLoader 
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu
from utils.get_palette import get_palette
from networks import U2NET


CATEGORY_MAPPING = {0: 'background', 1: 'upper body', 2: 'lower body', 3: 'full body'}

class ImageDataset(Dataset):
    def __init__(self, base_path, mapping_path):
        super(Dataset, self).__init__()
        self.base_path = base_path
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                Normalize_image(0.5, 0.5)
            ])
        self.mapping = pd.read_csv(mapping_path)

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):
        user_id = self.mapping["user_id"][index]
        look_id = self.mapping["look_id"][index]
        # user_id, look_id = list(map(int, self.paths[index].split("/")[-1][:-4].split("_")))
        img = Image.open("%s/%d_%d.jpg" % (self.base_path, user_id, look_id)).convert('RGB')
        img = img.resize((768, 768), Image.BICUBIC)
        img = self.transform(img)
        return user_id, look_id, img


class ClothSegmentation():

    def __init__(self, opt) -> None:
        self.opt = opt
        dataset = ImageDataset(self.opt.base_path, self.opt.mapping_path)
        self.n_images = len(dataset)
        print("There are %d images" % self.n_images)

        self.data_loader = DataLoader(
            dataset, 
            batch_size=self.opt.batch_size, 
            num_workers=self.opt.num_workers, shuffle=False, drop_last=False
        )
    
    def get_device(self):
        print("CUDA available: %s" % torch.cuda.is_available())
        if self.opt.use_cuda == -1:
            print("use cpu")
            device = torch.device("cpu")
        else:
            print("use cuda: %d" % self.opt.use_cuda)
            device = torch.device("cuda:%d" % (self.opt.use_cuda))
        return device

    def load_model(self, device, in_ch=3, out_ch=4):
        net = U2NET(in_ch=in_ch, out_ch=out_ch)
        net = load_checkpoint_mgpu(net, self.opt.model_path)
        net = net.to(device).eval()
        return net

    def get_cloth_segment(self, inp_img, palette_img, file_name): 
        """
        takes in segmented palette image and gives the mapping on raw image
        @param inp_img: raw input image resize to 768 * 768
        @param palette_img: image with palette 
        @param file_name: path to save output image
        """
        color_dict = {}
        cate_list = [0, 0, 0]

        # note down the pixel region, exclude 0: background
        for pixel in np.unique(palette_img):
            if pixel != 0:
                region = np.where(palette_img == pixel)
                cate_list[pixel-1] = 1
                color_dict[pixel] = list(zip(region[0], region[1]))
        
        # count the number of detected items
        segmentation_count = 0

        # assigning the segmented pixels form the dict to corresponding original image 
        for cate_key in color_dict.keys(): 
            
            # initializing a white image 
            segmented_image = np.full(inp_img.shape, 255) 
            segmentation_count += 1

            # mapping pixels into white image 
            for key in color_dict[cate_key]:
                segmented_image[key[0]][key[1]][0] = inp_img[key[0]][key[1]][0] 
                segmented_image[key[0]][key[1]][1] = inp_img[key[0]][key[1]][1]
                segmented_image[key[0]][key[1]][2] = inp_img[key[0]][key[1]][2]
            
            # convert from RGB to BGR, save by cv2
            segmented_image = segmented_image[:,:,::-1]
            cv2.imwrite(file_name + "_%d.jpg" % cate_key, segmented_image)

        return segmentation_count, cate_list

    def run(self):

        device = self.get_device()
        # limit cpu usage
        torch.set_num_threads(8)
        print("load U2NET model")
        net = self.load_model(device)
            
        palette = get_palette(4)
        batch_no = 0

        res = []
        for user_ids, look_ids, imgs in self.data_loader:
            if batch_no % self.opt.print_interval == 0:
                print("Batch %d / %d" % (batch_no, int(self.n_images / self.opt.batch_size)))

            output_tensor = net(imgs.to(device))
            # shape: (batch_size, category_cnt, width, height)
            output_tensor = F.log_softmax(output_tensor[0], dim=1)
            output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
            output_arr = output_tensor.cpu().numpy()

            for i in range(imgs.shape[0]):
                img_arr = np.squeeze(output_arr[i], axis=0)

                palette_img = Image.fromarray(img_arr.astype('uint8'), mode='L')
                palette_img.putpalette(palette)
                
                img = Image.open("%s/%d_%d.jpg" % (self.opt.base_path, user_ids[i], look_ids[i])).convert('RGB')
                img = img.resize((768, 768), Image.BICUBIC)

                segmentation_cnt, cate_list = self.get_cloth_segment(
                    np.array(img), np.array(palette_img), 
                    "%s/%d_%d" % (self.opt.save_path, user_ids[i], look_ids[i])
                )
                res.append([user_ids[i].item(), look_ids[i].item(), segmentation_cnt] + cate_list)    
                
            batch_no += 1

        res_df = pd.DataFrame(res, columns=["user_id", "look_id", "segmentation_cnt", "upper body", "lower body", "full body"])
        res_df.to_csv("data/segmentation_output.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", type=int, default=0, help="-1 not use cuda, otherwise is the cuda idx, from 0 to 3")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers")
    parser.add_argument("--print_interval", type=int, default=100, help="print progress per interval of batches")
    parser.add_argument("--base_path", type=str, help="Path to load human images")
    parser.add_argument("--mapping_path", type=str, help="Path to load mapping csv")
    parser.add_argument("--model_path", type=str, help="Path to load model checkpoint")
    parser.add_argument("--save_path", type=str, help="Path to save the segmented cloth images")
    opt = parser.parse_args()

    cloth_seg = ClothSegmentation(opt)
    cloth_seg.run()

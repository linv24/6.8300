from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from dataloader import create_dataloader, SpatialDataset
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
SAVE_DATA_DIRECTORY = "data/predicate_probe_data"

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    default_dataset_args = {
        "split": 'train',
        "predicate_dim": 30,
        "object_dim": 67,
        "data_path": 'data/c_0.9_c_0.1.json',
        "load_img": True,
        "data_aug_shift": False,
        "data_aug_color": False,
        "crop": True,
        "norm_data": False,
        "resize_mask": False,
        "trans_vec": [],
    }

    allowed_predicates = set([
        "in front of", # occlusion, depth
        "behind", # occlusion, depth
        "below", # relative position
        "to the left of (wrt you)", # relative position
        "to the right of (wrt you)", # relative position
        "to the side of", # relative position
        "on", # relative position
        "covering", # occlusion
        "inside", # occlusion
        "near", # depth
        "far from", # depth
    ])

    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()



    for split in ("train", "valid", "test"):
        split_dataset_args = default_dataset_args | {"split": split}
        split_data_loader, split_predicates, split_objects = create_dataloader(
            **split_dataset_args,
            num_workers=1,
            batch_size=16,
        )
        split_samples_dict = defaultdict(list)

        print(f"processing {split} split...")
        for batch_ix, batch in tqdm(enumerate(split_data_loader), total=len(split_data_loader)):
            # prepare inputs
            images = [image for image in batch["img_crop"]]
            subject_names = batch["subject"]["name"]
            object_names = batch["object"]["name"]
            predicate_names = batch["predicate"]["name"]
            depth_images = [depth_image.repeat(3, 1, 1) for depth_image in batch["depth_crop"]] # duplicate channel dim
            weights = batch["weight"]
            labels = batch["label"].long()

            # encode images
            image_inputs = processor(images=images, return_tensors="pt", do_rescale=False).to(device)
            depth_image_inputs = processor(images=depth_images, return_tensors="pt", do_rescale=False).to(device)
            with torch.no_grad():
                image_embs = model.get_image_features(**image_inputs)
                depth_image_embs = model.get_image_features(**depth_image_inputs)

            # encode text
            subject_inputs = processor(text=subject_names, return_tensors="pt", padding=True).to(device)
            object_inputs = processor(text=object_names, return_tensors="pt", padding=True).to(device)
            predicate_inputs = processor(text=predicate_names, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                subject_embs = model.get_text_features(**subject_inputs)
                object_embs = model.get_text_features(**object_inputs)
                predicate_embs = model.get_text_features(**predicate_inputs)

            # construct and sort samples
            for i in range(len(images)):
                if predicate_names[i] in allowed_predicates:
                    sample = {
                        "image_emb": image_embs[i],
                        "subject_emb": subject_embs[i],
                        "object_emb": object_embs[i],
                        "predicate_emb": predicate_embs[i],
                        "depth_emb": depth_image_embs[i],
                        "subject_name": subject_names[i],
                        "object_name": object_names[i],
                        "predicate_name": predicate_names[i],
                        "weight": weights[i],
                        "label": labels[i],
                    }
                    split_samples_dict[predicate_names[i]].append(sample)

        print(f"saving samples for {split} split...")
        print(f"total samples in {split} split: {sum(len(v) for v in split_samples_dict.values())}")
        for predicate in split_samples_dict:
            save_file_path = f"{SAVE_DATA_DIRECTORY}/{predicate}_{split}.pt"
            torch.save(split_samples_dict[predicate], save_file_path)



if __name__ == "__main__":
    main()
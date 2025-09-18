# MIT License

# Copyright (c) 2024 Erfan Shayegani 😈

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Jailbreak In Pieces 
# This code finds an equivalent adversarial image to a harmful target image.

from PIL import Image
import numpy as np
import torch.optim as optim
from sentence_transformers import util
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel, CLIPImageProcessor
import torch
import matplotlib.pyplot as plt
import pickle
import argparse


def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images



def main() -> None:
    # if you have CUDA or MPS, set it to the active device like this
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    pilimage = Image.open(args.image_path)


    model_id = "lip-vit-large-patch14-336"

    # we initialize a tokenizer, image processor, and the model itself
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id) # for processing text
    processor = CLIPProcessor.from_pretrained(model_id) # for processing image
    model = CLIPModel.from_pretrained(model_id).to(device)# for giving us the embeddings

    imgproc = CLIPImageProcessor.from_pretrained(model_id)
    imgproc.do_normalize = True

    image = imgproc(images=pilimage, return_tensors="pt")['pixel_values'].to(device) # target image
    # (torch.float32, torch.Size([1, 3, 336, 336]))
    # plt.imshow(image.squeeze(0).cpu().permute(1, 2, 0))
    img_emb = model.get_image_features(image) # target embedding
    # (torch.float32, torch.Size([1, 768]), True)
    
    random_img = torch.zeros_like(image) # random image

    random_img.requires_grad = True
    # plt.imshow(random_img.detach().cpu().squeeze(0).T)
    random_img_emb = model.get_image_features(random_img) # Adversarial Image embedding
    pdist = torch.nn.PairwiseDistance(p=2)

    # Create Gradient mask - Uncomment in case you want to optmize only a part of the image
    # gradient_mask = torch.zeros(3, 336, 336).to(device)
    # # gradient_mask[:, 0:120, :] = 1.0
    # gradient_mask[:, 0:160, 224:] = 1.0
    # # gradient_mask[:, 0, :] = 1.0
    # # n = 0
    # # while n < 336:
    # #     gradient_mask[:, n, :] = 1.0
    # #     n = n + 1
    # random_img.register_hook(lambda grad: grad.mul_(gradient_mask))
    # Set up optimizer
    optimizer = optim.Adam([random_img], lr=0.01) # tune the learning rate
    # use smaller learning rates for less visible noise
    # Training loop

    num_epochs = 5000
    loss_list = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        random_img_emb = model.get_image_features(random_img) # get the new embedding after the random_img gets updated.
        # Compute loss
        loss = pdist(random_img_emb, img_emb.detach()) 

        loss.backward()
        optimizer.step()
        random_img.data = torch.clamp(random_img.data, 0.0, 1.0)
        # Print loss for monitoring progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
        loss_list.append(loss.item())


    # Save the list to a file using pickle
    with open(args.save_loss_path, 'wb') as f:
        pickle.dump(loss_list, f)

    # we can still visualize the processed image
    # plt.imshow(random_img.squeeze(0).detach().cpu().T)

    # Convert tensor to the range (0, 255) and convert to NumPy array
    tensor = random_img.cpu()
    tensor = denormalize(tensor)
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8).numpy()
    tensor = tensor.squeeze()
    # Reshape tensor to [3, 336, 336]
    tensor = np.transpose(tensor, (1, 2, 0))
    # Create PIL Image object
    imagee= Image.fromarray(tensor)
    # Save image as JPEG
    imagee.save(args.save_img_path)

    return

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--image_path", type=str, default="./images/tips.png")
    parse.add_argument("--save_img_path", type=str, default="./images/L2_noNorm_clipgrad_tips_336_LR0_1-1_1.png")
    parse.add_argument("--save_loss_path", type=str, default="./images/loss_curve/tips.pkl")
    args = parse.parse_args()
    main()
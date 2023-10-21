import os
import glob
import tqdm
import argparse
from PIL import Image
import torch
import torchvision.transforms as T


def main(model_checkpoint, image_dir, mask_dir, output_dir, max_size=1000):

    generator_state_dict = torch.load(model_checkpoint)['G']

    if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
        from model.networks import Generator
    else:
        from model.networks_tf import Generator  

    use_cuda_if_available = True
    device = torch.device('cuda' if torch.cuda.is_available()
                          and use_cuda_if_available else 'cpu')

    # set up network
    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)

    generator_state_dict = torch.load(model_checkpoint)['G']
    generator.load_state_dict(generator_state_dict, strict=True)

    # fetch 
    images = glob.glob(os.path.join(image_dir, '**')) 
    print(f"Find {len(images)} images!")

    os.makedirs(output_dir, exist_ok=True)
    print(image_dir, mask_dir)
    pbar = tqdm.tqdm(images, desc=f"Running DeepFillv2 - {model_checkpoint}")
    for image_filepath in pbar:
        pbar.set_postfix({'image': image_filepath})
        image_filename = os.path.basename(image_filepath)
        mask_filename, _ = os.path.splitext(image_filename)
        mask_filename = f"{mask_filename}.mask.png"

        # show input image and mask
        mask_filepath = os.path.join(mask_dir, mask_filename)

        # load image and mask
        # https://note.nkmk.me/en/python-opencv-bgr-rgb-cvtcolor/
        image = Image.open(image_filepath).convert("RGB")
        mask = Image.open(mask_filepath).convert("RGB")
        
        # resize for computational efficiency
        image_shape = None
        if image.size[-1] > max_size or image.size[-2] > max_size:
            image_shape = image.size
            image = image.resize((max_size, max_size))
            mask = mask.resize((max_size, max_size))

        # prepare input
        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)

        _, h, w = image.shape
        grid = 8

        image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
        mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

        image = (image*2 - 1.).to(device)  # map image values to [-1, 1] range
        mask = (mask > 0.5).to(dtype=torch.float32,
                            device=device)  # 1.: masked 0.: unmasked

        image_masked = image * (1.-mask)  # mask image

        ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
        x = torch.cat([image_masked, ones_x, ones_x*mask],
                    dim=1)  # concatenate channels

        with torch.inference_mode():
            _, x_stage2 = generator(x, mask)

        # complete image
        image_inpainted = image * (1.-mask) + x_stage2 * mask

        # save inpainted image
        img_out = ((image_inpainted[0].permute(1, 2, 0) + 1)*127.5)
        img_out = img_out.to(device='cpu', dtype=torch.uint8)
        img_out = Image.fromarray(img_out.numpy())
        img_out = img_out.resize(image_shape) if image_shape else img_out

        output_filepath = os.path.join(output_dir, image_filename)
        img_out.save(output_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--model-checkpoint', type=str, default="pretrained/states_pt_places2.pth")
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--mask-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--max-size', type=str, default=1000)
    args = parser.parse_args()

    # calculate square
    main(args.model_checkpoint, args.image_dir, args.mask_dir, args.output_dir)

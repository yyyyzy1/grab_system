import argparse
from fastsam import FastSAM, FastSAMPrompt
import ast
import torch
from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy
import cv2
import numpy as np
import time
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM-x.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="../demo_data/aimkse/difficult/rgb/3.png", help="path to image file"
    )
    parser.add_argument(
        "--depth_img_path", type=str, default="../demo_data/aimkse/difficult/depth/3.png", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="../demo_data/aimkse/difficult/masks/", help="image save path"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    return parser.parse_args()


def main(args):
    # load model
    model = FastSAM(args.model_path)
    args.point_prompt = ast.literal_eval(args.point_prompt)
    args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
    args.point_label = ast.literal_eval(args.point_label)
    input = Image.open(args.img_path)
    input = input.convert("RGB")
    depth_map = np.array(Image.open(args.depth_img_path))  # Assuming depth image is grayscale

    everything_results = model(
        input,
        device=args.device,
        retina_masks=args.retina,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou
        )

    masks_tensor = everything_results[0].masks.data
    min_area = 2000  # Minimum area to consider
    max_area = 6000  # Maximum area to consider


    filtered_masks = []
    average_depths = []  # To store average depth values for each valid mask

    for i in range(masks_tensor.shape[0]):
        # Get the i-th mask
        mask = masks_tensor[i]

        # Calculate the area by counting non-zero elements (number of pixels with value 1)
        area = torch.sum(mask).item()

        # Check if the area is within the specified range
        if min_area <= area <= max_area:
            filtered_masks.append(mask)  # Append the mask to the filtered list

            # Convert mask tensor to numpy array
            mask_np = mask.cpu().numpy()

            # Calculate the average depth value within the mask
            mask_area_depth = depth_map[mask_np == 1]
            avg_depth = np.mean(mask_area_depth)
            average_depths.append(avg_depth)  # Store the average depth



    sorted_indices = np.argsort(average_depths)  # Sort descending

    # First maximum average depth mask
    max_index = sorted_indices[2]
    max_avg_depth = average_depths[max_index]
    max_avg_depth_mask = filtered_masks[max_index].cpu().numpy()

    # Second maximum average depth mask
    second_max_index = sorted_indices[3]
    second_max_avg_depth = average_depths[second_max_index]
    second_max_avg_depth_mask = filtered_masks[second_max_index].cpu().numpy()

    # Save the mask with the maximum average depth
    mask_image_max = np.zeros_like(max_avg_depth_mask, dtype=np.uint8)
    mask_image_max[max_avg_depth_mask == 1] = 255

    cv2.imwrite("../demo_data/aimkse/difficult/masks/max_depth.png", mask_image_max)


    # Save the mask with the second maximum average depth
    mask_image_second_max = np.zeros_like(second_max_avg_depth_mask, dtype=np.uint8)
    mask_image_second_max[second_max_avg_depth_mask == 1] = 255
    cv2.imwrite("../demo_data/aimkse/difficult/masks/max_second_depth.png", mask_image_second_max)

    bboxes = None
    points = None
    point_label = None
    prompt_process = FastSAMPrompt(input, everything_results, device=args.device)
    if args.box_prompt[0][2] != 0 and args.box_prompt[0][3] != 0:
            ann = prompt_process.box_prompt(bboxes=args.box_prompt)
            bboxes = args.box_prompt
    elif args.text_prompt != None:
        ann = prompt_process.text_prompt(text=args.text_prompt)
    elif args.point_prompt[0] != [0, 0]:
        ann = prompt_process.point_prompt(
            points=args.point_prompt, pointlabel=args.point_label
        )
        points = args.point_prompt
        point_label = args.point_label
    else:
        ann = prompt_process.everything_prompt()
    prompt_process.plot(
        annotations=ann,
        output_path=args.output+args.img_path.split("/")[-1],
        bboxes = bboxes,
        points = points,
        point_label = point_label,
        withContours=args.withContours,
        better_quality=args.better_quality,
    )




if __name__ == "__main__":
    args = parse_args()
    main(args)

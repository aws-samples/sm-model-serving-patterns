
import numpy as np
import colorsys
from skimage.measure import find_contours
import random
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon

def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    
    for line in labels:
        line = line.rstrip("\n")
        ids = line.split(',')
        label_map[int(ids[0])] = ids[2] 
        
    return label_map


def random_colors(N, bright=False):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """    
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.3):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="", 
                      score_thres=0.5, mask_thres=0.5,
                      figsize=(10, 10), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, framework='pytorch'):
    """
    boxes: [num_instance, (x1, y1, x2, y2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    score_thres: To return only objects whose score is greater than to a certain value in the detected result.
    mask_thres: Threshold for binarizing the mask image
    figsize: (optional) the size of the image
    show_mask, show_bbox: To show masks and bounding boxes or not    
    colors: (optional) An array or colors to use with each object
    framework: pytorch/mxnet
    """
    
    if framework == 'mxnet':
        boxes = boxes.asnumpy()
        masks = masks.asnumpy()
        scores = scores.asnumpy()        
    else:
        boxes = np.array(boxes)
        masks = np.array(masks)   
        scores = np.array(scores)        
    
    # Get only results that are above the threshold. Default threshold is 0.5. 
    scores = scores[scores > score_thres]
    # Number of instances
    N = len(scores)

    if not N:
        print("\n*** No instances to display *** \n")

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    masked_image = image.astype(np.uint32).copy()
    
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        x1, y1, x2, y2 = boxes[i]
            
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        #predicted_class = class_info[int(cls_pred)]
        label = class_names[int(class_id)]
        caption = "{} {:.3f}".format(label, score) if score else label
        ax.text(x1, y1, caption, color='w', verticalalignment='top',
                size=12, bbox={'color': color, 'pad': 0})           

        # Mask
        mask = (masks[:, :, i] > mask_thres) * 1
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)

        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
        
    #return masked_image

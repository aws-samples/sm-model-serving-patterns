def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    
    for line in labels:
        line = line.rstrip("\n")
        ids = line.split(',')
        label_map[int(ids[0])] = ids[2] 
        
    return label_map


def get_label_map_imagenet(label_file):
    label_map = {}
    with open(label_file, 'r') as f:
        for line in f:
            key, val = line.strip().split(':')
            label_map[key] = val.replace(',', '')
    return label_map


def delete_endpoint(client, endpoint_name):
    response = client.describe_endpoint_config(EndpointConfigName=endpoint_name)
    model_name = response['ProductionVariants'][0]['ModelName']

    client.delete_model(ModelName=model_name)    
    client.delete_endpoint(EndpointName=endpoint_name)
    client.delete_endpoint_config(EndpointConfigName=endpoint_name)    
    
    print(f'--- Deleted model: {model_name}')
    print(f'--- Deleted endpoint: {endpoint_name}')
    print(f'--- Deleted endpoint_config: {endpoint_name}')    
    
    
def plot_bbox(img_resized, bboxes, scores, cids, class_info, framework='pytorch', threshold=0.5):

    import numpy as np
    import random
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    
    if framework=='mxnet':
        img_np = img_resized.asnumpy()
        scores = scores.asnumpy()
        bboxes = bboxes.asnumpy()
        cids = cids.asnumpy()
    else:
        img_np = img_resized
        scores = np.array(scores)
        bboxes = np.array(bboxes)
        cids = np.array(cids)    

    # Get only results that are above the threshold. Default threshold is 0.5. 
    scores = scores[scores > threshold]
    num_detections = len(scores)
    bboxes = bboxes[:num_detections, :]
    cids = cids[:num_detections].astype('int').squeeze()

    # Get bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    random.seed(42)
    random.shuffle(colors)
    
    plt.figure()
    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.imshow(img_np)

    if cids is not None:
        # Get unique class labels 
        unique_labels = set(list(cids.astype('int').squeeze()))
        unique_labels = np.array(list(unique_labels))
        n_cls_preds = len(unique_labels)
        bbox_colors = colors[:n_cls_preds]

        for b, cls_pred, cls_conf in zip(bboxes, cids, scores):
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            predicted_class = class_info[int(cls_pred)]
            label = '{} {:.2f}'.format(predicted_class, cls_conf)
            
            # Get box height and width
            box_h = y2 - y1
            box_w = x2 - x1

            # Add a box with the color for this class
            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(bbox)

            plt.text(x1, y1, s=label, color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})

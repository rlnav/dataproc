import torch
from mmseg.apis import init_model, inference_model, show_result_pyplot

config_path = '/home/ruslan/PycharmProjects/WildScenes/wildscenes/configs/deeplabv3/deeplabv3_r50-d8_2xb20-80k_wildscenes-512x512_standard.py'
checkpoint_path = '/home/ruslan/PycharmProjects/WildScenes/pretrained_models/deeplabv3_wildscenes.pth'
img_path='/media/ruslan/VRAS-DATA 4TB 2/datasets/ROUGH/marv_2024-09-26-13-46-51/images/1727351423_104193449_camera_right.png'


model = init_model(config_path, checkpoint_path)
result = inference_model(model, img_path)
print(result.pred_sem_seg.data.shape, torch.unique(result.pred_sem_seg.data))

# display the segmentation result
# vis_image = show_result_pyplot(model, img_path, result, out_file='result.png')
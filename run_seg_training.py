import os
from nets.unet3d import *
from utils.network_training import network_training
import shutil
import hiddenlayer as hl

os.environ['CUDA_VISIBLE_DEVICES']='3'

input_path = '/home/local/PARTNERS/yl715/segmentationScratch/abc/inputs/'
output_path = '/home/local/PARTNERS/yl715/segmentationScratch/abc/outputs/'

output_image_path= output_path + '/images/'
output_model_path= output_path + 'model/'
this_file = '/home/local/PARTNERS/yl715/segmentationScratch/run_seg_training.py'

# Load data
image_train = np.load(input_path + 'input.npy')
label_train = np.load(input_path + 'mask.npy')
image_valid = None
label_valid = None

# Network architecture arguments
model_args = dict(
            input_channels = 1,
            base_num_features = 8,
            num_blocks_per_depth_encoder = (2, 2, 2, 2),
            feat_map_mul_on_downscale = 2,
            pool_layer_kernel_sizes = ((1, 1, 1),
                                    (1, 1, 1),
                                    (1, 1, 1),
                                    (1, 1, 1)),
                                    # (1, 1, 1)),
                                    #(2, 2, 2),
                                    #(2, 2, 2)),
                                    # ),
            conv_layer_kernel_sizes = ((3, 3, 3),
                                    (3, 3, 3),
                                    (3, 3, 3),
                                    (3, 3, 3)),
                                    # (3, 3, 3)),
                                    #),
                                    #(3, 3, 3),
                                    #(3, 3, 3)),
                                  # ),
            layers_config = get_layers_config(input_dim=3, dropout_p=None,nonlin="LeakyReLU", norm_type="InstanceNorm"),
            num_classes = 1,
            num_blocks_per_depth_decoder = (2,2,2),
            deep_supervision=False,
            upscale_logits=False,
            max_features=512,
            initializer=None
)

# Training arguments
train_args = dict(
    image_train = image_train,
    label_train = label_train,
    output_image_path = output_image_path,
    output_model_path = output_model_path,
    num_epoches=2,
    num_input_channels = 1,
    batch_size = 1,
    lr = 3e-5,
    momentum = 0.95,
)

unet =UNet(**model_args).cuda()

if output_path is not None and not os.path.exists(output_path):
    os.makedirs(output_path)
    os.makedirs(output_image_path)
    os.makedirs(output_model_path)

network_training(
    net=unet,
    **train_args
)
shutil.copy(this_file, output_path)


# activate this if you want to visulize the built network
# volume_size = (160, 128, 128)
# dummy_input = torch.rand((1, 1, *volume_size)).cuda()
# dummy_gt = (torch.rand((1, 1, *volume_size)) * 2).round().clamp_(0, 47).cuda().long()
# out= unet(dummy_input)
# s=hl.build_graph(unet, dummy_input)
# s.save("myNet3_hl.txt")





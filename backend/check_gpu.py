import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print(f"GPU found: {gpus}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"TF version: {tf.__version__}")

if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=3500)]
        )
        print("GPU configured successfully!")
    except RuntimeError as e:
        print(f"GPU config error: {e}")
else:
    print("NO GPU — training will use CPU only")

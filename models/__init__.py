
import simple_convnet, two_channel, siamese, spatial_transformer, matching

def simple_convnet_inference(locks, keys, eval=False, output_size=1):
    return simple_convnet.inference(locks, keys, eval, output_size)

def two_channel_inference(locks, keys, eval=False, output_size=1):
    return two_channel.inference(locks, keys, eval, output_size)

def siamese_inference(locks, keys, eval=False, output_size=1):
    return siamese.inference(locks, keys, eval, output_size)

def matching_inference(locks, keys, eval=False, output_size=1):
    return matching.inference(locks, keys, eval, output_size)

# Transformers
def rotate_and_translate_transformer(U, theta, out_size, name='rotate_and_translation_transformer'):
    return spatial_transformer.rotate_and_translate_transformer(U, theta, out_size, name)

def affine_transformer(U, theta, out_size, name='SpatialTransformer', **kwargs):
    return spatial_transformer.affine_transformer(U, theta, out_size, name)

import simple_model, two_channel, siamese

def simple_model_inference(locks, keys, eval=False):
    return simple_model.inference(locks, keys, eval)

def two_channel_inference(locks, keys, eval=False):
    return two_channel.inference(locks, keys, eval)

def siamese_inference(locks, keys, eval=False):
    return siamese.inference(locks, keys, eval)
from numpy.random import shuffle
from tensorflow import convert_to_tensor

def data_generator(images, caption_map, features_map, batch_size):
    '''
    Generates infinate data training/validation batches of size batch size for the list of images
    
    Code based of example from this tutorial: 
    - Source: A step-by-step guide to building an image caption generator using Tensorflow 
    - https://medium.com/@khaledeemam/a-step-by-step-guide-to-building-an-image-caption-generator-using-tensorflow-a9e0a87cc0cb 
    '''
    while True:
        features, input_seq, target_seq = [], [], []
        current_batch_size= 0
        
        shuffle(images)
        
        for image in images:
            if image in caption_map and image in features_map:
                for caption in caption_map[image]:
                    features.append(features_map[image])
                    input_seq.append(caption[:-1])
                    target_seq.append(caption[1:])
                    
                    current_batch_size += 1
                    if current_batch_size == batch_size: 
                        try:
                            yield [convert_to_tensor(features), convert_to_tensor(input_seq)], convert_to_tensor(target_seq)
                        except Exception as e:
                            print(f"Error: {e}")
                            raise
                        features, input_seq, target_seq = [], [], []
                        current_batch_size  = 0
        if features:
            yield (convert_to_tensor(features), convert_to_tensor(input_seq)), convert_to_tensor(target_seq) 

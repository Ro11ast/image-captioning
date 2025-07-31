from tensorflow import image, io, convert_to_tensor
from tensorflow.data import Dataset, AUTOTUNE # type: ignore
from tensorflow.keras.backend import clear_session # type: ignore
from os.path import join 
from keras.applications.inception_resnet_v2 import preprocess_input
import matplotlib.pyplot as plt

def display_image(directory_path, image_name):
    image_path = io.gfile.join( directory_path, image_name)
    image_data = io.read_file(image_path)
    image_data = image.decode_jpeg(image_data, channels=3)
    plt.imshow(image_data)
    plt.show()

def load_image(image_path):
    image_data = io.read_file(image_path)
    image_data = image.decode_jpeg(image_data, channels=3)
    image_data = image.resize(image_data, [224, 224])
    return image_data

def process_image(image_path):
    image_data = load_image(image_path)
    return preprocess_input(image_data)
        
def batch_extract_features(images, directory, encoder, batch_size = 1024):
    image_paths = [join(directory, image) for image in images]
    image_dataset = Dataset.from_tensor_slices(image_paths)
    image_dataset = image_dataset.map(process_image, num_parallel_calls=AUTOTUNE)
    image_dataset = image_dataset.batch(batch_size).prefetch(AUTOTUNE)
    
    feature_mappings = {}
    batch_num = 0
    
    for batch in image_dataset:
        extracted_features = encoder.predict(batch, verbose=1)
        batch_idx = batch_num * batch_size
        
        for i, features in enumerate(extracted_features):
            name = images[i + batch_idx]
            feature_mappings[name] = convert_to_tensor(features)
            
        batch_num += 1    
    return feature_mappings
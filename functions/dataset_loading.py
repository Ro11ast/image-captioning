from json import load
from numpy import fromstring, zeros

# Flicker 8k
'''
Flicker8K 

'''
def load_flicker8k_captions():
    filepath = 'datasets/flicker8k/Flickr8k.token.txt'
    with open(filepath) as file:
        caption_map = {}
        total_captions = []
        
        for line in file.readlines():
            line = line.rstrip("\n")
            image_name, caption = line.split('\t')
            image_name = image_name[:-2]
        
            if image_name not in caption_map:
                caption_map[image_name] = []
                
            caption_map[image_name].append(caption)
            total_captions.append(caption)
            
        return caption_map, total_captions

def load_flicker8k_split():
    train_path = 'datasets/flicker8k/Flickr_8k.trainImages.txt'
    val_path = 'datasets/flicker8k/Flickr_8k.devImages.txt'
    test_path = 'datasets/flicker8k/Flickr_8k.testImages.txt'
    
    with open(train_path) as file:
        train_images = file.read().strip().split('\n')
        
    with open(val_path) as file:
        val_images = file.read().strip().split('\n')
        
    with open(test_path) as file:
        test_images = file.read().strip().split('\n')
        
    return train_images, val_images, test_images

# COCO dataset
'''
MS COCO captions dataset: 
used in training, evaluating and testing models 

Source:
title -  Microsoft COCO: Common Objects in Context 
author - Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Dollár
year - 2015
url - https://arxiv.org/abs/1405.0312 

'''

def read_coco_subset(filepath): 
    with open(filepath) as file: 
        data = load(file)
        
        image_data = data['images']
        annotation_data = data['annotations']
        
        image_paths = {image['id']: image['file_name'] for image in image_data}
        image_captions = {}
        total_captions = []

        for annotation in annotation_data:
            image_id = annotation['image_id']
            caption = annotation['caption']
            
            if image_id not in image_captions:
                image_captions[image_id] = []
                
            image_captions[image_id].append(caption)
            total_captions.append(caption)
        
        dataset = {}

        for image_id, file_name in image_paths.items():
            if image_id in image_captions:                            
                dataset[file_name] = image_captions[image_id] 
        return dataset, total_captions
            
    
def load_coco_captions():
    train_path = 'datasets/coco/annotations/captions_train2014.json'
    val_path = 'datasets/coco/annotations/captions_val2014.json'
    
    train_caption_map, train_captions = read_coco_subset(train_path)
    val_caption_map, val_captions = read_coco_subset(val_path)
    
    return train_caption_map, val_caption_map, train_captions + val_captions

# Mapping functions
def map_captions_to_images(images, captions):
    caption_mappings = {}
    
    for i in range(len(images)):
        if images[i] not in caption_mappings: 
            caption_mappings[images[i]] = []
            
        caption_mappings[images[i]].append(captions[i])
    
    return caption_mappings

# Embeddings
'''
Code used for loading embedding and creating embedding matrix copied from: 
Using pre-trained word embeddings - https://keras.io/examples/nlp/pretrained_word_embeddings/
'''

def load_embeddings(emb_size):
    path_to_glove_file = 'datasets/glove/glove.6B.'+ str(emb_size) +'d.txt'

    embeddings_index = {}

    with open(path_to_glove_file, encoding="utf-8") as file:
        for line in file:
            word, coefs = line.split(maxsplit=1)
            coefs = fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors" % len(embeddings_index))
    return embeddings_index

def create_embedding_matrix(emb_size, voacb_size, word_to_idx):
    
    embeddings_index = load_embeddings(emb_size)
    
    hits = 0
    misses = 0

    embedding_matrix = zeros((voacb_size, emb_size))

    for word, index in word_to_idx.items():
        embedding_vector = embeddings_index.get(word)
        
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            hits += 1
        else:
            misses += 1
            
    print("Converted %d words (%d misses)" % (hits, misses))
    return embedding_matrix
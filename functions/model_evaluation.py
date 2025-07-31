from tensorflow import data, zeros, int32, float32, tensor_scatter_nd_add, constant, convert_to_tensor, math, expand_dims
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.meteor_score import meteor_score
from keras.utils import pad_sequences
from numpy import array, append, mean, argmax, argsort, log
from pickle import load

def decode_caption(caption, idx_to_word):
    decoded_caption = []
    
    for word in caption: 
        if word == 3 or word == 0:
            continue
        if word == 4: 
            break
        else: 
            token =  idx_to_word[word] 
            decoded_caption.append(token) 
            
    return decoded_caption

def decode_captions(captions, idx_to_word):
    decoded_captions = []
    
    for caption in captions:
        decoded_captions.append(decode_caption(caption, idx_to_word))
            
    return decoded_captions
    
def greedy_search_caption(model, image_features, idx_to_word, seq_len, start_token=3, end_token=4):
    image_features = expand_dims(image_features, axis= 0)
   
    caption = [start_token]
            
    for i in range(seq_len):  
        input_seq = pad_sequences([caption], maxlen=seq_len, padding='post')
        prediction = model.predict([image_features, input_seq], verbose = 0)[0]
        
        word_idx = argmax(prediction[i])
          
        caption.append(word_idx) 
        
        if word_idx == end_token:
            break
                
    return decode_caption(caption, idx_to_word)

def beam_search_caption(model, image_features, idx_to_word, seq_len, beam_width=3, temperature=2.0, start_token=3, end_token=4):
    image_features = expand_dims(image_features, axis=0)
    beams = [(array([start_token]), 0.0)]

    # Function for length penalty
    def length_penalty(length, alpha=0.7):
        return ((5 + length) / 6) ** alpha

    for i in range(seq_len - 1):
        all_candidates = []
        
        for seq, score in beams:
            if seq[-1] == end_token:
                all_candidates.append((seq, score))
                continue
            elif seq[-1] == end_token:
                continue
            
            input_seq = pad_sequences([seq], maxlen=seq_len, padding='post')
            predictions = model.predict([image_features, input_seq], verbose=0)[0]
            scaled_predictions = predictions[i] / temperature
            
            top_k_indices = argsort(scaled_predictions)[-beam_width:]
            
            for word_idx in top_k_indices:
                # Calculate log probability with numerical stability
                log_prob = log(scaled_predictions[word_idx] + 1e-10)
                
                # Calculate new score with length penalty
                new_seq = append(seq, word_idx)
                new_score = (score + log_prob) / length_penalty(len(new_seq))
                
                all_candidates.append((new_seq, new_score))
        
        beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        if all(seq[-1] == end_token for seq, _ in beams):
            break
        
    beam_captions = []
    
    for i in range(beam_width):
        if i < len(beams):
            beam_captions.append(decode_caption(beams[i][0], idx_to_word))
            
    return beam_captions

def generate_batch(model, features, seq_len, batch_size):
    captions = zeros([batch_size, seq_len], dtype=int32)
    start_token = constant([3] * batch_size, dtype=int32)
    indices = [[i, 0] for i in range(batch_size)]
    captions = tensor_scatter_nd_add(captions, indices, start_token)
    features = convert_to_tensor(features)
    
    for i in range(seq_len - 1):
        prediction = model.predict([features, captions], verbose=0)
        updates = constant(argmax(prediction[:, i, :], axis = 1), dtype=int32) 
        indices = constant([[j, i + 1] for j in range(batch_size)], dtype=int32)
        captions = tensor_scatter_nd_add(captions, indices, updates)
    return captions

def generate_captions(model, images, features_map, caption_map, idx_to_word, seq_len=20, batch_size=1024): 
    predicted_captions, true_captions = [], []
    current_features, current_images = [], []
    current_batch_size = 0
    
    for image in images:
        current_images.append(image)
        current_features.append(features_map[image])
        current_batch_size += 1
        
        if current_batch_size >= batch_size:
            batch_prediction = generate_batch(model, current_features, seq_len, batch_size).numpy()
            predicted_captions += [decode_caption(cap, idx_to_word) for cap in batch_prediction]
            
            for captions in [caption_map[image] for image in current_images]:
                true_captions.append([decode_caption(cap, idx_to_word) for cap in captions])
                
            current_features, current_images = [], []
            current_batch_size = 0
        
    # Dealing with leftover data if batch size isnt reached    
    if len(current_images) > 0: 
        batch_size = len(current_images)
        batch_prediction = generate_batch(model, current_features, seq_len, batch_size).numpy()
        predicted_captions += [decode_caption(cap, idx_to_word) for cap in batch_prediction]
        
        for captions in [caption_map[image] for image in current_images]:
            true_captions.append([decode_caption(cap, idx_to_word) for cap in captions])
            
    return predicted_captions, true_captions

def evaluate_captions(predicted_captions, true_captions):
    bleu1, bleu2, bleu3, bleu4, meteor= [], [], [], [], []
    bleu_weights = [(1.,), (1./2., 1./2.), (1./3., 1./3., 1./3.), (1./4., 1./4., 1./4., 1./4.)]
    smoothing_function = SmoothingFunction().method4
 
    for i in range(len(predicted_captions)):
        bleu_score = sentence_bleu(true_captions[i], predicted_captions[i], bleu_weights, smoothing_function)
        bleu1.append(bleu_score[0])
        bleu2.append(bleu_score[1])
        bleu3.append(bleu_score[2])
        bleu4.append(bleu_score[3])
        
        meteor.append(meteor_score(true_captions[i], predicted_captions[i]))
    
    bleu = [mean(bleu1), mean(bleu2), mean(bleu3), mean(bleu4)]
    meteor = mean(meteor)
    return bleu, meteor

def generate_and_evaluate_caption(decoder, image_features, target_captions, idx_to_word, seq_len):
    greedy_caption = greedy_search_caption(decoder, image_features, idx_to_word, seq_len)
    beam_captions = beam_search_caption(decoder, image_features, idx_to_word, seq_len)
    
    true_captions = []
    
    for caption in target_captions: 
        decoded_caption = decode_caption(caption, idx_to_word)
        true_captions.append(decoded_caption)
        
    bleu_smoothing_function = SmoothingFunction().method4 
    blue_weights = [(1.,), (1./2., 1./2.), (1./3., 1./3., 1./3.), (1./4., 1./4., 1./4., 1./4.)]

    greedy_bleu = sentence_bleu(true_captions, greedy_caption, blue_weights, bleu_smoothing_function)
    greey_meteor = meteor_score(true_captions, greedy_caption)
    
    best_caption = []
    best_bleu = []
    best_meteor = float('-inf')
    
    for caption in beam_captions:
        beam_bleu = sentence_bleu(true_captions, caption, blue_weights, bleu_smoothing_function)
        beam_meteor = meteor_score(true_captions, caption)
        if best_meteor < beam_meteor:
            best_caption = caption
            best_bleu = beam_bleu
            best_meteor = beam_meteor

    return (greedy_caption, greedy_bleu, greey_meteor), (best_caption, best_bleu, best_meteor), true_captions


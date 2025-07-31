from tensorflow import strings
from nltk import download
from nltk.corpus import stopwords
download('stopwords')

def remove_stop_words(text):
    STOP_WORDS = set(stopwords.words('english'))
    text = text.lower()
    words = text.split()
    filtered_words = [word for word in words if word not in STOP_WORDS]
    text = " ".join(filtered_words)
    return text

def custom_standardization(text):
    text = strings.lower(text)
    text = strings.regex_replace(text, r"[!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~]?", "")
    text = strings.regex_replace(text, r'[0-9]', '')
    text = strings.join(['<START> ', text, ' <END>'])
    return text

def create_vocab_mappings(vocab):
    word_to_idx, idx_to_word = {}, {}
    for i, word in enumerate(vocab):
        word_to_idx[word] =  i
        idx_to_word[i] = word 
    return word_to_idx, idx_to_word

def vectorize_captions(caption_map, vectorizer, captions_per_image=5):
    images = list(caption_map.keys())
    vectorized_map = {}

    # Loop through each set of captions 
    for i in range(captions_per_image):
        batch = [caption_map[img][i] for img in images]
        
        # Vectorize the batch of captions
        vectorized_batch = vectorizer(batch).numpy()
        
        # Map the image back to vectorized captions
        for img, vectorized_caption in zip(images, vectorized_batch):
            if img not in vectorized_map:
                vectorized_map[img] = []
            vectorized_map[img].append(vectorized_caption)

    return vectorized_map
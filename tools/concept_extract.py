import json
import argparse
import sng_parser
from tqdm import tqdm
from collections import defaultdict

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_file", type=str, default="train_image_info.json")
    parser.add_argument("--output_file", type=str, default="tags.json")
    args = parser.parse_args()

    data = json.load(open(args.ann_file, 'r'))
    images = data['images']
    NN = defaultdict(list)
    parser = sng_parser.Parser('spacy', model='en_core_web_trf')

    for img in tqdm(images):
        id = str(img['id'])
        for cap in img['captions']:
            graph = parser.parse(cap.lower())
            for entity in graph['entities']:
                word = entity['lemma_head']
                if word in NN.keys():
                    NN[word].append(id)
                elif len(word.split(' ')) > 1:
                    NN[word].append(id)
                elif wn.synsets(word):
                    NN[word].append(id)

    with open(args.output_file, 'w') as nn:
        json.dump(NN, nn)

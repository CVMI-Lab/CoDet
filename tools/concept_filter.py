import json
import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict


def non_object(word):
    wn_l = wn.synsets(word)
    wn_l = wn_l if len(wn_l) < 2 else wn_l[:2]
    for word in wn_l:
        while word not in neg_set:
            if word in pos_set:
                return False
            tmp = word.hypernyms()
            if tmp == []:
                break
            else:
                word = tmp[0]
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_file", type=str, default="train_image_info.json")
    parser.add_argument("--tag_file", type=str, default="tags.json")
    parser.add_argument("--output_file", type=str, default="train_image_info_tags.json")
    args = parser.parse_args()

    root = wn.synsets('entity')
    obj = wn.synsets('object')
    person = wn.synsets('person')
    people = wn.synsets('people')
    food = wn.synsets('food')
    location = wn.synsets('location')
    photograph = wn.synsets('photograph')
    photo = wn.synsets('photo')
    picture = wn.synsets('picture')
    system = wn.synsets('system')
    land = wn.synsets('land')
    structure = wn.synsets('structure')

    pos_set = set(obj + food)
    neg_set = set(root + location + photo + photograph + picture + system + land + structure)

    cc_data = json.load(open(args.anno_file, 'r'))
    cat2img = json.load(open(args.tag_file, 'r'))

    cat2id = dict()
    img2id = defaultdict(list)

    for k in list(cat2img.keys()):
        if ' ' in k.strip():
            deleted = False
            last_word = k.strip().split(' ')[-1]
            for word in k.strip().split(' '):
                if not wn.synsets(word):
                    del cat2img[k]
                    deleted = True
                    break
            if not deleted and non_object(last_word):
                del cat2img[k]
        else:
            if not wn.synsets(k.strip()):
                del cat2img[k]
            elif non_object(k.strip()):
                del cat2img[k]

    new_cat2img = defaultdict(list)
    for k, v in cat2img.items():
        k = k.strip().replace('/ ', '/').replace(' ', '_')
        new_cat2img[k] = new_cat2img[k] + v

    # filter out concepts with frequency less than 20
    for k, v in list(new_cat2img.items()):
        if len(v) < 20:
            del new_cat2img[k]

    # update category info
    new_cat_info = []
    for i, cat in enumerate(sorted(new_cat2img.keys())):
        cat2id[cat] = i+1
        new_cat_info.append({'id': i+1, 'name': cat, 'synonyms': [cat], 'frequency': 'f', 'supercategory': cat})

    # add image label to each image
    for k, v in new_cat2img.items():
        for img in v:
            img2id[int(img)].append(cat2id[k])
    images = []
    for i, x in enumerate(cc_data['images']):
        if x['id'] in img2id:
            x['pos_category_ids'] = img2id[x['id']]
            images.append(x)

    # write to output
    out_data = {'images': images, 'categories': new_cat_info, 'annotations': []}
    for k, v in out_data.items():
        print(k, len(v))
    json.dump(out_data, open(args.output_file, 'w'))









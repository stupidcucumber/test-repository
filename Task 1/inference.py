import torch
import argparse
from bert import BertModel
from transformers import BertTokenizerFast
import nltk
import json


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_path', type=str,
                        help='Path to the folder with information about model')
    parser.add_argument('-i', '--input', type=str,
                        help='Path to the file with input.')
    parser.add_argument('--output_format', type=str, default='txt',
                        help='Choose between TXT, HTML formats. Only TXT is supported right now.')
    parser.add_argument('-o', '--output', type=str, default='inference.json',
                        help='Name of the file where output will be stored.')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='On which device to run the model.')
    parser.add_argument('--threshold', type=float, default=0,
                        help='Defines the threhold for probability.')

    return parser.parse_args()


def inference_sentence_tags(model: BertModel, tokenizer, sentence: str, threshold: float, device: str='cpu'):
    '''
        This function is bugged, because it do not take into consideration variant that it will be:

        "Mountain Appalashia" -> tokens([Mountain, Appa, lash, ia]) -> classes([B-mount, I-mount, I-mount, I-mount])

        But when we will use this function in the result we will get 4 tokens instead of 2.

        Now I fixed it, but it must be tested!
    '''
    model.eval()
    decoder_dict = {
        0: 'B-mount',
        1: 'I-mount',
        2: 'O'
    }
    data = tokenizer(sentence, padding='max_length',
                                            max_length=512, truncation=True,
                                            return_tensors="pt")
    word_ids = data.word_ids()

    input_id = data['input_ids'].to(device)
    mask = data['attention_mask'].to(device)

    logits = model(input_id, mask, None)
    logits = logits[0][0]
    
    indexes = []
    previous_id = None
    for index, id in enumerate(word_ids):
        current_id = logits[index].argmax()
        if id is None:
            continue
        elif id != previous_id:
            indexes.append(current_id)
            previous_id = id
        else:
            continue
            
    result = [decoder_dict[index.item()] for index in indexes]
    print(result)

    return result


def highlight_mountians(sentence, tags):
    result = ''
    begin = '[\BEGIN]'
    end = '[\END]'

    mountain = False
    for index, token in enumerate(nltk.word_tokenize(sentence)):
        if tags[index] == 'B-mount':
            result += begin + ' ' + token + ' '
            mountain = True
        elif tags[index] == 'O' and mountain == True:
            result += end + ' ' + token + ' '
            mountain =False
        else:
            result += token + ' '

    if mountain:
        result += end

    return result


def divide_to_sentences(text: str):
    result = nltk.sent_tokenize(text)

    return result


if __name__ == '__main__':
    args = parse_arguments()

    DEVICE = args.device

    model = BertModel(weight_path=args.model_path).to(DEVICE)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    
    with open(args.input, 'r') as f:
        text = f.read()

    result = [highlight_mountians(sentence, inference_sentence_tags(model=model, 
                                      tokenizer=tokenizer,
                                      sentence=sentence,
                                      device=DEVICE)) for sentence in divide_to_sentences(text)]

    sentences = {
        'sentences' : result
    }

    with open(args.output, '+w') as f:
        f.write(json.dumps(sentences, indent=2))

    print('Inference is over!')

from transformers import BertTokenizerFast


class Tokenizer():
    def __init__(self):
        pass

    def tokenize(self, sentence, tags):
        raise NotImplementedError('This funtion must be overrided by the child class!')


class BertTokenizer(Tokenizer):
    '''
        This tokenizer takes sentence and tokenizes it, producing tokens,
    attention mask and labels itself.
    '''
    def __init__(self, vocabulary: str=None, tags_list: list=None):
        if vocabulary is None:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        else:
            self.tokenizer = BertTokenizerFast(vocabulary)

        self.dictionary_tags = {
            value: index for index, value in enumerate(tags_list)
        }


    def _align_tags(self, tokenized_sentence, dataset_tags):
        '''
            If we have None in word_id, then its index will be set as -100.
        '''

        word_ids = tokenized_sentence.word_ids() # The id's of the words the current sub-word is being part of in the original sentence.
        result = []

        for word_id in word_ids:
            if word_id is None:
                result.append(-100)
            else:
                try:
                    result.append(self.dictionary_tags[dataset_tags[word_id]])
                except:
                    result.append(-100)


        return result

    
    def _pad_labels(self, labels: list, expected_length: int=512, pad: str='O'):
        while len(labels) != expected_length:
            labels.append(pad)

        return labels


    def tokenize(self, sentence, tags):
        max = 512
        tokenized_sentence = self.tokenizer(sentence, padding='max_length',
                                            max_length=max, truncation=True,
                                            return_tensors="pt")
        aligned_ids = self._align_tags(tokenized_sentence, tags)

        return {
            'labels' : aligned_ids, 
            'attention_mask' : tokenized_sentence['attention_mask'],
            'input_ids' : tokenized_sentence['input_ids']
        }
import os
from matplotlib.image import imread
import numpy as np
import pandas as pd
from tensorflow.contrib import learn
import fasttext
import numpy as np
from sklearn.utils import shuffle
import re


class PreProcessing:
    def __init__(self,data_src):

        self.similar_pairs = self.build_corpus('./data_repository/questions.csv')
        self.embeddings_model = fasttext.train_unsupervised("./data_repository/text_corpus.txt", model='skipgram', lr=0.005, dim=64,
                                            ws=5, epoch=200)
        self.embeddings_model.save_model("./model_siamese_network/ft_skipgram_ws5_dim64.bin")
        print('FastText training finished successfully.')
        self.current_index = 0
        input_X = list(self.similar_pairs['question1'])
        input_Y = list(self.similar_pairs['question2'])
        wc_list_x = list(len(x.split(' ')) for x in input_X)
        wc_list_y = list(len(x.split(' ')) for x in input_Y)
        wc_list = []
        wc_list.extend(wc_list_x)
        wc_list.extend(wc_list_y)
        max_document_length = 16                 # or use a constant like 16, select this parameter based on your understanding of what could be a good choice
        number_of_elements = len(input_X)
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        full_corpus = []
        full_corpus.extend(input_X)
        full_corpus.extend(input_Y)
        full_data = np.asarray(list(self.vocab_processor.fit_transform(full_corpus)))
        self.embeddings_lookup = []
        for word in list(self.vocab_processor.vocabulary_._mapping):
            try:
                self.embeddings_lookup.append(self.embeddings_model[str(word)])
            except:
                pass
        self.embeddings_lookup = np.asarray(self.embeddings_lookup)
        self.vocab_processor.save('./model_siamese_network/vocab')
        self.write_metadata(os.path.join('model_siamese_network','metadata.tsv'),list(self.vocab_processor.vocabulary_._mapping))
        print('Vocab processor executed and saved successfully.')
        self.X = full_data[0:number_of_elements]
        self.Y = full_data[number_of_elements:2*number_of_elements]
        self.label = list(self.similar_pairs['is_duplicate'])

    def preprocess(self,x):
        try:
            tk_x = x.lower()

            # list of characters which needs to be replaced with space
            space_replace_chars = ['?', ':', ',', '"', '[', ']', '~', '*', ';', '!', '?', '(', ')', '{', '}', '@', '$',
                                   '#', '.', '-', '/']
            tk_x = tk_x.translate({ord(x): ' ' for x in space_replace_chars})

            non_space_replace_chars = ["'"]
            tk_x = tk_x.translate({ord(x): '' for x in non_space_replace_chars})

            # remove non-ASCII chars
            tk_x = ''.join([c if ord(c) < 128 else '' for c in tk_x])

            # replace all consecutive spaces with one space
            tk_x = re.sub('\s+', ' ', tk_x).strip()

            # find all consecutive numbers present in the word, first converted numbers to * to prevent conflicts while replacing with numbers
            regex = re.compile(r'([\d])')
            tk_x = regex.sub('*', tk_x)
            nos = re.findall(r'([\*]+)', tk_x)
            # replace the numbers with the corresponding count like 123 by 3
            for no in nos:
                tk_x = tk_x.replace(no, "<NUMBER>", 1)

            return tk_x.strip().lower()
        except:
            return ""

    def build_corpus(self, filepath):
        similar_items = pd.read_csv(filepath)
        selected_cols = ['question1', 'question2', 'is_duplicate']
        similar_items = similar_items[selected_cols]
        similar_items['question1'] = similar_items['question1'].apply(self.preprocess)
        similar_items['question2'] = similar_items['question2'].apply(self.preprocess)
        similar_items = shuffle(similar_items)
        similar_items = similar_items.drop_duplicates()
        question_list = list(similar_items['question1'])
        question_list.extend(list(similar_items['question2']))
        pd.DataFrame(question_list).to_csv('./data_repository/text_corpus.txt', index=False)
        print('Text corpus generated and persisted successfully.')
        return similar_items

    def write_metadata(self,filename, labels):
        with open(filename, 'w') as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(labels):
                f.write("{}\t{}\n".format(index, label))

        print('Metadata file saved in {}'.format(filename))

    def get_siamese_batch(self, n):
        last_index = self.current_index
        self.current_index += n
        return self.X[last_index: self.current_index, :], self.Y[last_index: self.current_index, :], np.expand_dims(self.label[last_index: self.current_index], axis=1)


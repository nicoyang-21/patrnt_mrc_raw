import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import json
import logging
import os
from sklearn.model_selection import train_test_split
import collections
import six
from transformers import BertTokenizer


def pad_sequence(sequences, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None是，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=1)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=0)
    return out_tensors


def cache(func):
    """
    本修饰器的作用是将SQuAD数据集中data_process()方法处理后的结果进行缓存，下次使用时可直接载入！
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        filepath = kwargs['filepath']
        postfix = kwargs['postfix']
        data_path = filepath.split('.')[0] + '_' + postfix + '.pt'
        if not os.path.exists(data_path):
            logging.info(f"缓存文件 {data_path} 不存在，重新处理并缓存！")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            logging.info(f"缓存文件 {data_path} 存在，直接载入缓存文件！")
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper


class LoadSingleSentenceClassificationDataset:
    pass


class LoadQADataset(LoadSingleSentenceClassificationDataset):
    """
        Args:
            doc_stride: When splitting up a long document into chunks, how much stride to
                        take between chunks.
                        当上下文过长时，按滑动窗口进行移动，doc_stride表示每次移动的距离
            max_query_length: The maximum number of tokens for the question. Questions longer than
                        this will be truncated to this length.
                        限定问题的最大长度，过长时截断
            n_best_size: 对预测出的答案近后处理时，选取的候选答案数量
            max_answer_length: 在对候选进行筛选时，对答案最大长度的限制

        """

    def __init__(self, doc_stride=64,
                 max_query_length=64,
                 n_best_size=20,
                 max_answer_length=30,
                 tokenizer=None,
                 **kwargs):
        super(LoadQADataset, self).__init__(**kwargs)
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.tokenizer = tokenizer
        self.max_sen_len = 512
        self.SEP_IDX = 102
        self.PAD_IDX = 0
        self.batch_size = 16

    @staticmethod
    def get_format_text_and_word_offset(text):
        """
        格式化原始输入的文本（去除多个空格）,同时得到每个字符所属的元素（单词）的位置
        这样，根据原始数据集中所给出的起始index(answer_start)就能立马判定它在列表中的位置。
        :param text:
        :return:
        e.g.
            text = "Architecturally, the school has a Catholic character. "
            return:['Architecturally,', 'the', 'school', 'has', 'a', 'Catholic', 'character.'],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3,
             3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        """

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            else:
                return False

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        # 以下这个for循环的作用就是将原始context中的内容进行格式化
        for c in text:  # 遍历paragraph中的每个字符
            if is_whitespace(c):  # 判断当前字符是否为空格（各类空格）
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:  # 如果前一个字符是空格
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c  # 在list的最后一个元素中继续追加字符
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        return doc_tokens, char_to_word_offset

    def preprocess(self, file_path, is_train=True):
        """
        将原始数据进行预处理，同时返回得到答案在原始context中的具体开始和结束位置（以单词为单位）
        :param file_path: JSON文件路径
        :param is_train: 是否用于训练
        :return:
        返回形式为一个二维列表，内层列表中的各个元素分别为 ['问题ID','原始问题文本','答案文本','context文本',
        '答案在context中的开始位置','答案在context中的结束位置']，并且二维列表中的一个元素称之为一个example,即一个example由六部分组成
        如下示例所示：
        [['5733be284776f41900661182', 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
        'Saint Bernadette Soubirous', 'Architecturally, the school has a Catholic character......',
        90, 92],
         ['5733be284776f4190066117f', ....]]
        """
        with open(file_path) as f:
            raw_data = json.load(f)
            data = raw_data["data"]
        examples = []
        for i in tqdm(range(len(data)), ncols=80, desc="正在便利json文件段落"):
            paragraphs = data[i]["paragraphs"]
            for j in range(len(paragraphs)):
                context = paragraphs[j]["context"]
                context_tokens, word_offset = self.get_format_text_and_word_offset(context)
                qas = paragraphs[j]["qas"]
                for k in range(len(qas)):
                    question_text = qas[k]["question"]
                    qas_id = qas[k]["id"]
                    if is_train:
                        answer_offset = qas[k]["answers"][0]["answer_start"]
                        orig_answer_text = qas[k]["answers"][0]["text"]
                        answer_len = len(orig_answer_text)
                        start_position = word_offset[answer_offset]
                        end_position = word_offset[answer_offset + answer_len - 1]
                        actual_text = " ".join(context_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(orig_answer_text.strip().split())
                        if actual_text.find(cleaned_answer_text) == -1:
                            print(f"Could not find answer: {actual_text} vs. {cleaned_answer_text}")
                            continue
                    else:
                        start_position = None
                        end_position = None
                        orig_answer_text = None
                    examples.append([qas_id, question_text, orig_answer_text, " ".join(context_tokens),
                                     start_position, end_position])
        return examples

    @staticmethod
    def improve_answer_span(self, context_tokens, answer_tokens, start_position, end_position):
        """
        本方法的作用有两个：
        1. 如https://github.com/google-research/bert中run_squad.py里的_improve_answer_span()函数一样，
        用于提取得到更加匹配答案的起止位置；
        2. 根据原始起止位置，提取得到token id中答案的起止位置
        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.

        context = "The leader was John Smith (1895-1943).
        answer_text = "1985"
        :param context_tokens: ['the', 'leader', 'was', 'john', 'smith', '(', '1895', '-', '1943', ')', '.']
        :param answer_tokens: ['1895']
        :param start_position: 5
        :param end_position: 5
        :return: [6,6]
        再例如：
        context = "Virgin mary reputedly appeared to Saint Bernadette Soubirous in 1858"
        answer_text = "Saint Bernadette Soubirous"
        :param context_tokens: ['virgin', 'mary', 'reputed', '##ly', 'appeared', 'to', 'saint', 'bern', '##ade',
                                        '##tte', 'so', '##ub', '##iro', '##us', 'in', '1858']
        :param answer_tokens: ['saint', 'bern', '##ade', '##tte', 'so', '##ub', '##iro', '##us'
        :param start_position = 5
        :param end_position = 7
        return (6,13)

        """
        new_end = None
        for i in range(start_position, len(context_tokens)):
            if context_tokens[i] != answer_tokens[0]:
                continue
            for j in range(len(answer_tokens)):
                if answer_tokens[j] != context_tokens[i + j]:
                    break
                new_end = i + j
            if new_end - i + 1 == len(answer_tokens):
                return i, new_end
        return start_position, end_position

    def data_process(self, file_path, is_train=True, postfix='cache'):
        """
        :param filepath:
        :param is_training:
        :return: [[example_id, feature_id, input_ids, seg, start_position,
                 end_position, answer_text, example[0]],input_tokens,token_to_orig_map [],[],[]...]
                分别对应：[原始样本Id,训练特征id,input_ids，seg，开始，结束，答案文本，问题id,input_tokens,token_to_orig_map]
        """
        examples = self.preprocess(file_path, is_train)
        all_data = []
        example_id, feature_id = 0, 1000000000
        # 由于采用了滑动窗口，所以一个example可能构造得到多个训练样本（即这里被称为feature）；
        # 因此，需要对其分别进行编号，并且这主要是用在预测后的结果后处理当中，训练时用不到
        # 当然，这里只使用feature_id即可，因为每个example其实对应的就是一个问题，所以问题ID和example_id本质上是一样的
        for example in tqdm(examples, ncols=80, desc="处理examples"):
            question_ids = self.tokenizer(example[1])["input_ids"]
            question_tokens = self.tokenizer.convert_ids_to_tokens(question_ids)
            context_ids = self.tokenizer(example[3])["input_ids"].drop(0)
            start_position, end_position, answer_text = -1, -1, None
            if is_train:
                start_position, end_position = example[4], example[5]
                answer_text = example[2]
                answer_tokens = self.tokenizer(answer_text)
            rest_len = self.max_sen_len - len(question_ids) - 1
            context_ids_len = len(context_ids)
            print(f"## 上下文长度为：{context_ids_len}, 剩余长度 rest_len 为 ： {rest_len}")
            if context_ids_len > rest_len:
                print("## 进入滑动窗口 …… ")
                s_idx, e_idx = 0, rest_len
                while True:
                    # We can have documents that are longer than the maximum sequence length.
                    # To deal with this we do a sliding window approach, where we take chunks
                    # of the up to our max length with a stride of `doc_stride`.
                    tmp_context_ids = context_ids[s_idx: e_idx]
                    tmp_context_tokens = self.tokenizer.convert_ids_to_tokens(tmp_context_ids)
                    input_ids = torch.tensor(question_ids + tmp_context_ids)
                    input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                    seg = [0] * len(question_ids) + [1] * (len(input_ids) - len(question_ids))
                    seg = torch.tensor(seg)
                    if is_train:
                        new_start_position, new_end_position = 0, 0
                        if start_position >= s_idx and end_position and end_position <= e_idx:
                            print("划窗中存在答案")
                            new_start_position = start_position - s_idx
                            new_end_position = new_start_position + (end_position - start_position)

                            new_start_position += len(question_ids)
                            new_end_position += len(question_ids)
                        all_data.append([example_id, feature_id, input_ids, seg, new_start_position, new_end_position,
                                         answer_text, example[0], input_tokens])
                    else:
                        all_data.append([example_id, feature_id, input_ids, seg, start_position,
                                         end_position, answer_text, example[0], input_tokens])
                    feature_id += 1
                    if e_idx >= context_ids_len:
                        break
                    s_idx += self.doc_stride
                    e_idx += self.doc_stride
            else:
                input_ids = torch.tensor(question_ids + context_ids)
                input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                seg = [0] * len(question_ids) + [1] * (len(input_ids) - len(question_ids))
                seg = torch.tensor(seg)
                if is_train:
                    start_position += len(question_ids)
                    end_position += len(question_ids)
                all_data.append([example_id, feature_id, input_ids, seg, start_position, end_position,
                                 answer_text, example[0], input_tokens])
            example_id += 1
            data = {"all_data": all_data, "max_len": self.max_sen_len, "example": examples}
            return data

    def generate_batch(self, data_batch):
        batch_input, batch_seg, batch_label, batch_qid = [], [], [], []
        batch_example_id, batch_feature_id, batch_map = [], [], []
        for item in data_batch:
            # item: [原始样本Id,训练特征id,input_ids，seg，开始，结束，答案文本，问题id,input_tokens,ori_map]
            batch_example_id.append(item[0])  # 原始样本Id
            batch_feature_id.append(item[1])  # 训练特征id
            batch_input.append(item[2])  # input_ids
            batch_seg.append(item[3])  # seg
            batch_label.append([item[4], item[5]])  # 开始, 结束
            batch_qid.append(item[7])  # 问题id
            # batch_map.append(item[9])  # ori_map
        batch_input = pad_sequence(batch_input, padding_value=self.PAD_IDX, max_len=self.max_sen_len)
        batch_seg = pad_sequence(batch_seg, padding_value=self.PAD_IDX, max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_input, batch_seg, batch_label, batch_qid, batch_example_id, batch_feature_id

    def load_train_val_test_data(self, train_file_path=None, val_file_path=None, test_file_path=None, only_tet=False):
        doc_stride = str(self.doc_stride)
        max_sen_len = str(self.max_sen_len)
        max_query_length = str(self.max_query_length)
        postfix = doc_stride + '_' + max_sen_len + '_' + max_query_length
        data = self.data_process(file_path=test_file_path,
                                 is_train=False,
                                 postfix=postfix)
        test_data, example = data["all_data"], data["example"]
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if only_tet:
            print(f"### 测试集{len(test_iter.dataset)}个")

        data = self.data_process(file_path=train_file_path,
                                 is_train=True, postfix=postfix)
        train_data, max_seq_len = data["all_data"], data["max_len"]
        _, val_data = train_test_split(train_data, test_size=0.2, random_state=2023)
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,  # 构造DataLoader
                              shuffle=False, collate_fn=self.generate_batch)
        print(f"## 训练集样本（{len(train_iter.dataset)}）个、开发集样本（{len(val_iter.dataset)}）个"
              f"测试集样本（{len(test_iter.dataset)}）个.")
        return train_iter, test_iter, val_iter

    @staticmethod
    def get_best_indexes(logits, n_best_size):
        """Get the n-best logits from a list."""
        # logits = [0.37203778 0.48594432 0.81051651 0.07998148 0.93529721 0.0476721
        #  0.15275263 0.98202781 0.07813079 0.85410559]
        # n_best_size = 4
        # return [7, 4, 9, 2]
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def get_final_text(self, pred_text, orig_text):
        pass

from transformers import BertModel, BertTokenizer, BertConfig
import torch.nn as nn


class BertforQA(nn.Module):
    """
    构建QA模型
    """

    def __init__(self, config):
        super(BertforQA, self).__init__()
        self.bert = BertModel.from_pretrained(config.model_path)
        self.dropout = nn.Dropout(0.2)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids,
                attention_mask,
                start_positions=None,
                end_positions=None):
        """
        :param input_ids: [batch_size,src_len]
        :param attention_mask: [batch_size,src_len]
        :param token_type_ids:
        :param position_ids:
        :param start_positions: [batch_size]
        :param end_positions: [batch_size]
        :return:
        """

        outputs = self.bert(input_ids, attention_mask)
        outputs = outputs["last_hidden_state"]
        logits = self.qa_outputs(outputs)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeezq(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # 由于部分情况下start/end 位置会超过输入的长度
            # （例如输入序列的可能大于512，并且正确的开始或者结束符就在512之后）
            # 那么此时就要进行特殊处理
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fn = nn.CrossEntropyLoss(ignore_index=ignored_index)
            # 这里指定ignored_index其实就是为了忽略掉超过输入序列长度的（起始结束）位置
            # 在预测时所带来的损失，因为这些位置并不能算是模型预测错误的（只能看做是没有预测），
            # 同时如果不加ignore_index的话，那么可能会影响模型在正常情况下的语义理解能力
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)

            return (start_loss + end_loss) / 2, start_logits, end_logits
        else:
            return start_logits, end_logits

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("./bert_model")
    tem = tokenizer("i love you '[SEP]' you", max_length=40, padding="max_length", truncation=True)
    print(tokenizer.convert_ids_to_tokens([101, 1045, 2293, 2017, 1005, 102, 1005, 2017, 102]))
    print(tem)





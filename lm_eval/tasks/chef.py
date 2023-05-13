
import lm_eval.datasets.coqa.coqa
from lm_eval.base import MultipleChoiceTask
import numpy as np


class CHEF(MultipleChoiceTask):
    VERSION=0.1
    DATASET_PATH = "lm_eval/datasets/chef"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = "chef_cn"

    def has_training_docs(self):
        # TODO: Fill in the return with `True` if the Task has training data; else `False`.
        return True

    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return True

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # TODO: Return the training document generator from `self.dataset`.
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            # TODO: Return the validation document generator from `self.dataset`.
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            # TODO: Return the test document generator from `self.dataset`.
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        print("working process")
        # TODO: Process the documents into a dictionary with the following keys:
        return {
            "query": "通过以下证据：{}判断声明的正确性{}（正确为0，错误为2，无法判断为1）".format(doc["evidence1"]+';'+doc["evidence2"],doc["claim"]),  # The claim prompt.
            # "query":doc["claim"],
            "choices": ["0","1","2"],  # The list of choices.
            "gold": doc["label"],  # The integer used to index into the correct element of `"choices"`.
        }

    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        return doc["query"]
    
    # def doc_to_target(self, doc):
    #     # TODO: Format the query prompt portion of the document example.
    #     return doc["label"]
    # def test_docs(self):
    #     dataset = self.dataset["test"] #通过self.dataset["validation"]调用原始数据形式
    #         """
    #         现在dataset对象为原始数据形式，为一个元素为字典的list
    #         该方法返回一个元素为字典的list，其中每个字典由原始字典数据转化为如下格式
    #         {
    #         "query": 模型输入prompt,
    #         "choices": <一个元素为str的list，表示候选项>,
    #         "gold": 正确答案在"choices"列表里的索引,
    #         }
    #         例如：
    #         {
    #         "query": "在地球上太阳每天从什么地方升起，是东边，西边还是南边？",
    #         "choices": ["东边","西边","南边"],
    #         "gold": 0,
    #         }
    #         """
    #         return new_dataset
    # def doc_to_text(self, doc):
    #         #参数doc为一个字典，为转换后的格式
    #         #该函数用于对输入prompt做最后修改，若无需修改，可直接返回doc["query"]
    #         return doc["query"]

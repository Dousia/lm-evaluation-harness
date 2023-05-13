from lm_eval.base import MultipleChoiceTask
from random import randint


class Mandarinograd(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "lm_eval/datasets/mandarinograd"
    DATASET_NAME = "mandarinograd"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        pass

    def validation_docs(self):
        pass

    def test_docs(self):
        dataset = self.dataset["test"]
        new_dataset = []
        for doc in dataset:
            roll = randint(0, 1)
            new_doc = {
                "query": "现有文本：{}\n问题：{}\n请选择以下两个选项中的一个作为回答：\n1.{}\n2.{}\n".format(doc["text"], doc['question'], doc['correct_answer'], doc['wrong_answer']),
                "choices": [doc['correct_answer'], doc['wrong_answer']] if roll == 0 else [doc['wrong_answer'], doc['correct_answer']],
                "gold": roll
            }
            new_dataset.append(new_doc)
        return new_dataset

    def _process_doc(self, doc):
        return doc

    def doc_to_text(self, doc):
        return doc['query']


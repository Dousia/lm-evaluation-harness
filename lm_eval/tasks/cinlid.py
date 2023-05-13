
from lm_eval.base import MultipleChoiceTask


class CINLID(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "lm_eval/datasets/CINLID"
    DATASET_NAME = "CINLID"

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
        if self.has_test_docs():
            dateset = self.dataset["test"]
            new_dataset = []
            for doc in dateset:
                new_doc = {
                    "query": "推理下面两个短语之间的关系：{}\t{}\n请选择以下选项中的一个作为回答：\n1.neutral\n2.entailment\n3.contradiction".format(doc["phrase1"],doc["phrase2"]),
                    "choices": ["neutral","entailment","contradiction"],
                    "gold": ["neutral","entailment","contradiction"].index(doc["label"])
                }
                new_dataset.append(new_doc)
            return new_dataset

    def _process_doc(self, doc):
        return doc

    def doc_to_text(self, doc):
        return doc["query"]

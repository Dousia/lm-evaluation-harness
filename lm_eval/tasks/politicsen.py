# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""
from lm_eval.base import MultipleChoiceTask

# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""


# TODO: Replace `NewTask` with the name of your Task.
class PoliticSen(MultipleChoiceTask):
    VERSION = 0.1
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "lm_eval/datasets/politicsen"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = "politicsen"

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
            # TODO: Return the test document generator from `self.dataset`.
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            dataset = self.dataset["test"]
            new_dataset = []
            for topic, label, text, chat, hit in zip(dataset["topic"], dataset["label"], dataset["answer"], dataset["Chatgpt"], dataset["HITGPT"]):
                new_dataset.append(
                    {
                        "label": label,
                        "TEXT": text
                    }
                )
            return map(self._process_doc, new_dataset)

    def _process_doc(self, doc):
        choices = [
            "句子不存在政治敏感内容",
            "存在明显或轻微政治敏感内容"
        ]
        out_doc = {
            "query": doc["TEXT"],
            "choices": choices,
            "gold": doc["label"]
        }
        return out_doc

    def doc_to_text(self, doc):
        prompt = "现有文本：“{0}”\n请判断该文本属于下面两个选项中的哪一个：\n1.{1}。\n2.{2}。".format(doc['query'], doc['choices'][0], doc['choices'][1])
        return prompt

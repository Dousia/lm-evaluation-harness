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
class Pclue_dev_nli(MultipleChoiceTask):
    VERSION = 0.1
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "lm_eval/datasets/pclue_dev_nli"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = "pclue_dev_nli"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return  True

    def training_docs(self):
        pass

    def validation_docs(self):
        pass
    
    def test_docs(self):
        dataset =  self.dataset["test"]
        return map(self._process_doc, dataset)

    def _process_doc(self, doc):
        out_doc = {
            "query": doc["input"],
            "choices": doc["answer_choices"],
            "gold": doc["answer_choices"].index(doc["target"]),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    




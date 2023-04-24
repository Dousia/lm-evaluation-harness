
"""
CoQA: A Conversational Question Answering Challenge
https://arxiv.org/pdf/1808.07042.pdf

CoQA is a large-scale dataset for building Conversational Question Answering
systems. The goal of the CoQA challenge is to measure the ability of machines to
understand a text passage and answer a series of interconnected questions that
appear in a conversation.

Homepage: https://stanfordnlp.github.io/coqa/
"""
import inspect
import transformers.data.metrics.squad_metrics as squad_metrics
from lm_eval.base import Task, rf, mean
from itertools import zip_longest
import os
import sys

class ChnSentiCorp(Task):
    VERSION = 0
    file_path = os.path.abspath(__file__)
    dir_path = file_path[:file_path.rfind('/')]
    DATASET_PATH = os.path.join(dir_path, '..', 'datasets', 'chnsenticorp')
    DATASET_NAME = 'chnsenticorp'

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
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return "{}\nQuestion: Is this sentence positive or negative?\nAnswer:".format(
            doc["text"],
        )
        # return doc['text']

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["story"] + " " + "\n".join(doc["questions"]["input_text"])

    @classmethod
    def get_answers(cls, doc, turn_id):
        # Returns unique answers and valid alternatives (Some questions in CoQA have multiple valid answers).
        answers = []
        answer_forturn = doc["answers"]["input_text"][turn_id - 1]
        answers.append(answer_forturn)

        additional_answers = doc.get("additional_answers")
        if additional_answers:
            for key in additional_answers:
                additional_answer_for_turn = additional_answers[key]["input_text"][
                    turn_id - 1
                ]
                if additional_answer_for_turn.lower() not in map(str.lower, answers):
                    answers.append(additional_answer_for_turn)
        return answers


    def doc_to_target(self, doc, turnid=None):
        return " {}".format({'pos': "positive", 'neg': "negative"}[doc["label"]])
        # Default to prediction of last turn.
        # if turnid is None:
        #     turnid = len(doc["questions"]["input_text"])
        # raw_text = doc["answers"]["input_text"][turnid - 1]
        # return " " + raw_text

    def construct_requests(self, doc, ctx):
        ll_positive, _ = rf.loglikelihood(ctx, " positive")
        ll_negative, _ = rf.loglikelihood(ctx, " negative")
        return ll_positive, ll_negative


    def process_results(self, doc, results):
        ll_positive, ll_negative = results
        pred = ll_positive > ll_negative
        gold = {'pos': 0, 'neg': 1}[doc["label"]]
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


















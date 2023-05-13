from lm_eval.base import MultipleChoiceTask


class Chinese_hotel(MultipleChoiceTask):
    VERSION = 1.0
    DATASET_PATH = "lm_eval/datasets/chinese_hotel"  # 填写数据集地址，注意数据存储格式
    DATASET_NAME = "chinese_hotel"  # 填写数据集名字

    def has_training_docs(self):
        # 是否有训练集，若有，返回True
        return True

    def has_validation_docs(self):
        # 是否有验证集，若有，返回True
        return False

    def has_test_docs(self):
        # 是否有测试集，若有，返回True
        return True

    # 多选任务主要在这三个函数上实现修改
    # 普通任务的training/validation/validation_docs()函数无严格要求
    # 多选任务的training/validation/validation_docs()函数需要对原始数据处理为固定选择形式

    def training_docs(self):
        # 若有，通过self.dataset["train"]调用原始数据形式
        dataset = self.dataset["train"]
        new_dataset = []
        for item in dataset:
            new_dt = {
                "query": "现有文本：“{0}”\n请判断该文本对{1}表达的情感极性属于下面三个选项中的哪一个：\n1.消极。\n2.中性。\n3.积极。".format(item['text'], item['target']),
                "choices": ["消极", "中性", "积极"],
                "gold": item['label'] + 1,
            }
            new_dataset.append(new_dt)
        return new_dataset

    def validation_docs(self):
        # 若有，通过self.dataset["validation"]调用原始数据形式
        pass

    def test_docs(self):
        dataset = self.dataset["test"]
        new_dataset = []
        for item in dataset:
            new_dt = {
                "query": "现有文本：“{0}”\n请判断该文本对{1}表达的情感极性属于下面三个选项中的哪一个：\n1.消极。\n2.中性。\n3.积极。".format(item['text'], item['target']),
                "choices": ["消极", "中性", "积极"],
                "gold": item['label'] + 1,
            }
            new_dataset.append(new_dt)
        return new_dataset

    def doc_to_text(self, doc):
        # 参数doc为一个字典，为转换后的格式
        # 该函数用于对输入prompt做最后修改，若无需修改，可直接返回doc["query"]
        return doc["query"]
        # 上述train/validation/test_docs()函数可利用_process_doc函数实现统一处理，减少重复代码

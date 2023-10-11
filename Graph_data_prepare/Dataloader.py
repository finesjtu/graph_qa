import json

class Dataloder:
    def __init__(self, dataset_name):
        # def super(Dataloder, self).super()
        self.dataset_name = dataset_name

        self.data_path = '/root/NLPCODE/GAS/GTP/dataset_roberta/'
        self.train_data_path = self.data_path + self.dataset_name + '_train.json'
        self.test_data_path = self.data_path + self.dataset_name + '_test.json'
        self.dev_data_path = self.data_path + self.dataset_name + '_dev.json'
        self.data, self.label, self.uid, self.train_split, self.test_split, self.dev_split = self.raw_data_integration()

        # {'data': all_data, 'label': all_label, 'uid': all_uid, 'train_split': train_split, 'test_split': test_split,
        #  'dev_split': dev_split

    def json_load(self, path):
        with open(path, 'r') as f:
            data_list = []
            label_list = []
            uid_list = []
            raw = f.readlines()
            for i in raw:
                raw_data = json.loads(i)
                hypothesis = raw_data['hypothesis']
                premise = raw_data['premise']
                label = raw_data['label']
                uid = raw_data['uid']
                data_list.append(hypothesis + '. ' + premise)
                # data_list.append(hypothesis)
                label_list.append(label)
                uid_list.append(uid)

        return {'data_list':data_list, 'label_list':label_list, 'uid_list':uid_list}

    def raw_data_integration(self):
        train_data = self.json_load(self.train_data_path)
        test_data = self.json_load(self.test_data_path)
        dev_data = self.json_load(self.dev_data_path)

        train_length = len(train_data['label_list'])
        test_length = len(test_data['label_list'])
        dev_length = len(dev_data['label_list'])


        all_data = train_data['data_list'] + test_data['data_list'] + dev_data['data_list']
        all_label = train_data['label_list'] + test_data['label_list'] + dev_data['label_list']
        train_uid = train_data['uid_list']
        test_uid = [x + train_length for x in test_data['uid_list']]
        dev_uid = [x + train_length + test_length for x in dev_data['uid_list']]
        train_split = train_length - 1
        test_split = train_length + test_length - 1
        dev_split = train_length + test_length + dev_length - 1
        # print(train_uid)
        # print(test_uid)
        # print(dev_uid)
        # print(train_split, test_split, dev_split)
        all_uid = train_uid + test_uid + dev_uid

        # return {'data': all_data, 'label': all_label, 'uid': all_uid, 'train_split': train_split, 'test_split': test_split,
        #         'dev_split': dev_split}

        return all_data, all_label, all_uid, train_split, test_split, dev_split





if __name__ == '__main__':
    data = Dataloder('ibmcs')
    print(data.data)

import logging
import numpy as np
import pandas as pd
import time
import csv
import os
import torch
import torch.nn.functional as F
import deeplearn as dl
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
import config
from modules import LeNet, CNN, SimpleCNN, FusedModel
from record_split import RecordSplit
from utils import DataStore
import pickle
from multiprocessing import Pool
from time import sleep
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import PyTorchClassifier
from art.utils import compute_success
from foolbox.distances import LpDistance
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import torchvision.models as models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

class FP32PyTorchClassifier(PyTorchClassifier):
    def predict(self, x, batch_size=128, **kwargs):
        x = np.asarray(x, dtype=np.float32)  # این خط کلیدی برای جلوگیری از float64
        return super().predict(x, batch_size=batch_size, **kwargs)

class ForceFloat32(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def query_num(self):
        base = self.model.module if hasattr(self.model, "module") else self.model
        return getattr(base, "query_num", 0)

    @query_num.setter
    def query_num(self, v):
        base = self.model.module if hasattr(self.model, "module") else self.model
        setattr(base, "query_num", v)

    def forward(self, x):
        return self.model(x.float())

class Action:
    def __init__(self, args):
        self.logger = logging.getLogger('action')
        self.args = args
        self.dataset_name = args.dataset_name
        self.load_data()

    def load_data(self):
        self.logger.info(f'loading data ({self.dataset_name})')
        self.data_store = DataStore(self.args)
        self.save_name = self.data_store.save_name
        self.df, self.num_records, self.num_classes = self.data_store.load_raw_data()
        self.data_store.create_basic_folders()
        self.logger.info('loaded data')

    def split_records(self):
        split_para = self.num_records
        self.record_split = RecordSplit(split_para, args=self.args)
        self.record_split.split_shadow_target()
        self.record_split.sample_records(self.args.unlearning_method)
        self.data_store.save_record_split(self.record_split)

    def _determine_model(self):
        if self.dataset_name == 'mnist':
            return LeNet(self.args)
        elif self.dataset_name in ['cifar10', 'cifar100', 'stl10']:
            return CNN(self.dataset_name, self.args)
        else:
            raise Exception(f"Invalid dataset: No {self.dataset_name}, please choose in ['mnist', 'cifar10', 'stl10', "
                            f"'cifar100']")

def _maybe_hsj(x_np: np.ndarray, y_true: int, clf: PyTorchClassifier, atk: HopSkipJump):
    # y_pred from model (label-only logic)
    y_pred = int(np.argmax(clf.predict(x_np), axis=1)[0])
    if y_pred != int(y_true):
    # already misclassified => perturbation is 0 (Algorithm 1)
        return x_np.copy()
    return atk.generate(x=x_np)

class ActionModelTrain(Action):
    def __init__(self, args):
        super(ActionModelTrain, self).__init__(args)

    def split_records(self):
        split_para = self.num_records
        self.record_split = RecordSplit(split_para, args=self.args)
        self.record_split.split_shadow_target()
        self.record_split.sample_records(self.args.unlearning_method)
        self.data_store.save_record_split(self.record_split)

    def train_shadow_model(self):
        self.logger.info(f'training shadow model in {self.args.dataset_name}\n')
        path = config.SHADOW_MODEL_PATH + self.save_name + '/'
        self.data_store.create_folder(path)
        self.train_models(self.args.shadow_set_num, self.args.shadow_num_shard, path, 'shadow')
        self.logger.info(f'trained shadow model in {self.args.dataset_name}\n')

    def train_target_model(self):
        self.logger.info(f'training target model in {self.args.dataset_name}\n')
        path = config.TARGET_MODEL_PATH + self.save_name + '/'
        self.data_store.create_folder(path)
        self.train_models(self.args.target_set_num, self.args.target_num_shard, path, 'target')
        self.logger.info(f'trained target model in {self.args.dataset_name}\n')

    def _train_model(self, sample_set_indices, save_name, model_type, i, j):
        self.logger.info('training start, set %s, unlearning %s' % (i, j))
        original_model = self._determine_model()
        original_model = nn.DataParallel(original_model)
        if self.dataset_name in ['mnist', 'cifar10', 'cifar100', 'stl10']:
            train_dataset = Subset(self.df, sample_set_indices)
            train_loader = DataLoader(dataset=train_dataset, batch_size=int(self.args.batch_size[0]), shuffle=True)
            test_dataset = Subset(self.df, np.random.randint(0, len(self.df), 256))
            test_loader = DataLoader(dataset=test_dataset, batch_size=int(self.args.batch_size[0]), shuffle=True)
            dl.train_model(self.args, original_model, train_loader, model_type, save_name)
        else:
            raise Exception(f'Invalid dataset: No {self.dataset_name}')
        self.logger.debug('training finished, set %s, unlearning %s' % (i, j))

    def train_models(self, num_shadows, num_shard, save_path, model_type):
        pass


class ActionModelTrainScratch(ActionModelTrain):
    def __init__(self, args):
        super(ActionModelTrainScratch, self).__init__(args)
        self.logger = logging.getLogger('action_model_train')
        if self.args.is_sample:
            self.split_records()
        self.train_shadow_model()
        self.train_target_model()

    def train_models(self, num_shadows, num_shard, save_path, model_type):
        if not self.args.is_sample:
            self.record_split = pickle.load(open(config.SPLIT_INDICES_PATH + self.save_name, 'rb'))

        #  data split
        self.record_split.generate_sample(model_type)

        if self.args.is_train_multiprocess:
            p = Pool(50, maxtasksperchild=1)
        import psutil
        ps = psutil.Process()
        cores = ps.cpu_affinity()
        ps.cpu_affinity((cores[0:50]))

        for sample_index in range(num_shadows):

            sample_set = self.record_split.sample_set[sample_index]
            sample_indices = sample_set['set_indices']
            unlearning_set = sample_set['unlearning_set']
            save_name_original = save_path + 'original_S' + str(sample_index)

            self._train_model(sample_indices, save_name_original, model_type, sample_index, j=-1)
            for unlearning_set_index, unlearning_indices in unlearning_set.items():

                self.logger.debug("training %s model: sample set %s | unlearning set %s" % (
                model_type, sample_index, unlearning_set_index))

                case = "deletion"
                unlearning_train_indices = np.setdiff1d(sample_indices, unlearning_indices)

                save_name_unlearning = save_path + "_".join(
                    ("unlearning_S" + str(sample_index), str(unlearning_set_index)))
                self._train_model(unlearning_train_indices, save_name_unlearning, model_type, sample_index,
                                      unlearning_set_index)
            if self.args.is_train_multiprocess:
                p.close()
                p.join()

class ActionModelTrainSisa(ActionModelTrain):
    def __init__(self, args):
        super(ActionModelTrainSisa, self).__init__(args)
        self.logger = logging.getLogger('action_model_train')
        if self.args.is_sample:
            self.split_records()
        self.train_shadow_model()
        self.train_target_model()

    def train_models(self, num_sample, num_shard, save_path, model_type):

        self.record_split = pickle.load(open(config.SPLIT_INDICES_PATH + self.save_name, 'rb'))
        self.record_split.generate_sample(model_type)

        for i in range(num_sample):
            sample_set = self.record_split.sample_set[i]
            shard_set = sample_set["shard_set"]
            unlearning_indices = sample_set["unlearning_indices"]
            unlearning_shard_mapping = sample_set["unlearning_shard_mapping"]

            unlearning_indices = unlearning_indices[:self.args.shadow_unlearning_num]

            # train original model
            for j in range(num_shard):
                save_name = save_path + "original_S%s_M%s" % (i, j)
                self._train_model(shard_set[j], save_name, model_type, i, j)

            # train unlearning models
            for j in unlearning_indices:
                self.logger.debug("training %s model set %s unlearning %s" % (model_type, i, j))
                shard_index = unlearning_shard_mapping[j]
                shard_indices = shard_set[shard_index]
                indices = np.delete(shard_indices, np.where(shard_indices == j)[0])
                save_name_unlearning = save_path + "unlearning_S%s_M%s" % (i, shard_index) + "_" + str(j)
                self._train_model(indices, save_name_unlearning, model_type, i, j)


class ActionAttack(Action):
    def __init__(self, args):
        super(ActionAttack, self).__init__(args)
        self.logger = logging.getLogger('action_attack')
        self.load_split_record()
        self.attack_path = os.path.join(config.ATTACK_DATA_PATH, self.save_name)
        self.data_store.create_folder(self.attack_path)

    def load_split_record(self):
        self.record_split = self.data_store.load_record_split()

    def _get_clip_values(self):
    # clip range MUST match the normalized input space
        if self.dataset_name == "mnist":
            mean = (0.1307,)
            std = (0.3081,)
        elif self.dataset_name in ["cifar10", "cifar100"]:
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif self.dataset_name == "stl10":
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        else:
            return (0.0, 1.0)

        mins = [(0.0 - m) / s for m, s in zip(mean, std)]
        maxs = [(1.0 - m) / s for m, s in zip(mean, std)]
        return (float(min(mins)), float(max(maxs)))


    def obtain_distences(self, distance_type, name):
        import eagerpy as ep
        dataset = pd.DataFrame(
            columns=['original-original_model', 'original-unlearning_model', 'original_model-unlearning_model',
                     'label'])
        if distance_type == 3:
            dis = LpDistance(ep.inf)
        elif distance_type in [0, 1, 2]:
            dis = LpDistance(distance_type)
        else:
            raise Exception('')
        if name == 'shadow':
            df = self.train_adv_ex_df
        elif name == 'target':
            df = self.test_adv_ex_df
        else:
            raise Exception('')
        for i in range(len(df)):
            row = df.iloc[i]
            dataset.loc[len(dataset)] = [
                float(dis(row.iloc[0], row.iloc[1])[0]),
                float(dis(row.iloc[0], row.iloc[2])[0]),
                float(dis(row.iloc[1], row.iloc[2])[0]),
                int(row.iloc[-1]),
            ]

        print(name, "d1 stats:", dataset['original-original_model'].min(), dataset['original-original_model'].max())
        print(name, "d2 stats:", dataset['original-unlearning_model'].min(), dataset['original-unlearning_model'].max())
        print(name, "d3 stats:", dataset['original_model-unlearning_model'].min(), dataset['original_model-unlearning_model'].max())

        return dataset

    def _generate_test_case(self, index):
        case = None
        label = None
        import random
        r = 0
        try:
            r = random.randint(0, len(index)-1)
        except TypeError as e:
            index = [index]

        if self.dataset_name in ['mnist', 'cifar10', 'cifar100', 'stl10']:
            loader = DataLoader(Subset(self.df, index), batch_size=1)
            for _, (case, label) in enumerate(loader):
                if r == 0:
                    return case, label.item()
                else:
                    r -= 1
        else:
            raise Exception(f'Invalid test dataset: {self.dataset_name}')

    def launch_attack(self):
        self._train_attacker()
        self._test_attacker()

    def obtain_adversarial_example(self, name):
        pass

    def _train_attacker(self):
        self.record_split.generate_sample('shadow')
        attack_train_dataset = self.obtain_distences(self.args.distance_type, 'shadow')
        torch.save(attack_train_dataset, os.path.join(self.attack_path, 'attack_train_dataset'))
        train_dataset = pd.DataFrame(columns=['dis1', 'dis2', 'dis3'])
        for i in range(len(attack_train_dataset)):
            train_dataset.loc[i] = [attack_train_dataset.iloc[i, 0], attack_train_dataset.iloc[i, 1],
                                    attack_train_dataset.iloc[i, 2]]
        train_dataset = F.softmax(torch.tensor(np.array(train_dataset)), dim=1)
        self.clf = svm.SVC()  # (C=0.8, kernel='rbf', gamma=10, decision_function_shape='ovr')
        self.clf.fit(attack_train_dataset.iloc[:, 0:3], attack_train_dataset.iloc[:, -1])

        self.logistic = LogisticRegression(random_state=0, solver='lbfgs', max_iter=400, multi_class='ovr', n_jobs=1)
        self.logistic.fit(attack_train_dataset.iloc[:, 0:3], attack_train_dataset.iloc[:, -1])
        torch.save(self.logistic, os.path.join(self.attack_path, f'{self.dataset_name}_attack_model'))

        self.attack = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
        self.attack.fit(attack_train_dataset.iloc[:, 0:3], attack_train_dataset.iloc[:, -1])

    def _test_attacker(self):
        self.record_split.generate_sample('target')
        attack_test_dataset = self.obtain_distences(self.args.distance_type, 'target')
        torch.save(attack_test_dataset, os.path.join(self.attack_path, 'attack_test_dataset'))
        test_dataset = pd.DataFrame(columns=['dis1', 'dis2', 'dis3'])
        true_label = []
        for i in range(len(attack_test_dataset)):
            test_dataset.loc[i] = [attack_test_dataset.iloc[i, 0], attack_test_dataset.iloc[i, 1],
                                   attack_test_dataset.iloc[i, 2]]
            true_label.append(attack_test_dataset.iloc[i, -1])

        pred = self.logistic.predict_proba(attack_test_dataset.iloc[:, 0:3])
        label = torch.tensor(pred).max(1, keepdim=True)[1]
        correct = label.eq(torch.tensor(true_label).view_as(label)).sum().item()
        accuracy = correct / len(true_label)
        score = pred[:, 1]
        fpr, tpr, _ = roc_curve(true_label, score)

        from sklearn.metrics import auc
        attack_auc = auc(fpr, tpr)
        print("\n" + "="*30)
        print(f"  ATTACK RESULTS ({self.args.unlearning_method.upper()})")
        print("="*30)
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC Score: {attack_auc:.4f}")
        print("="*30 + "\n")



class ActionAttackScratch(ActionAttack):
    def __init__(self, args):
        super(ActionAttackScratch, self).__init__(args)

        self.train_adv_ex_df = self.obtain_adversarial_example('shadow')  # HopSkipJump
        torch.save(self.train_adv_ex_df, os.path.join(self.attack_path, 'train_adv_ex_df'))
        self.test_adv_ex_df = self.obtain_adversarial_example('target')
        torch.save(self.test_adv_ex_df, os.path.join(self.attack_path, 'test_adv_ex_df'))

        #self.train_adv_ex_df = torch.load(self.attack_path + self.dataset_name + '/train_adv_ex_df')
        #self.test_adv_ex_df = torch.load(self.attack_path + self.dataset_name + '/test_adv_ex_df')
        #self.train_dataset = torch.load(self.attack_path + self.dataset_name + '/attack_train_dataset')
        #self.test_dataset = torch.load(self.attack_path + self.dataset_name + '/attack_test_dataset')
        attack_train_dataset = self.obtain_distences(self.args.distance_type, 'shadow')
        attack_test_dataset = self.obtain_distences(self.args.distance_type, 'target')

        self.launch_attack()
    
    def obtain_adversarial_example(self, name):
        n = 0
        self.record_split.generate_sample(name)
        adv_ex_df = pd.DataFrame(columns=['original', 'original model', 'unlearning model', 'mem or non-mem'])

        # --- CSV time logging setup
        csv_path = os.path.join(os.path.join(self.attack_path, f"time_log_{name}_scratch.csv"))
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["sample_index", "unlearning_set_index", "label", "model_type", "elapsed_time_sec", "query_count"])

        if name == 'target':
            save_path = config.TARGET_MODEL_PATH + self.save_name + '/'
            set_num = self.args.target_set_num
        else:
            save_path = config.SHADOW_MODEL_PATH + self.save_name + '/'
            set_num = self.args.shadow_set_num

        for sample_index in range(set_num):
            sample_set = self.record_split.sample_set[sample_index]
            if 'unlearning_set' in sample_set:
                unlearning_set = sample_set['unlearning_set']
            elif 'unlearning_indices' in sample_set:
                unlearning_indices = sample_set['unlearning_indices']
                unlearning_set = {0: unlearning_indices} 
            else:
                continue

            save_name_original = save_path + 'original_S' + str(sample_index)

            model_original = torch.load(save_name_original, weights_only=False)
            model_original = ForceFloat32(model_original).float().eval()  

            model_original.query_num = 0  

            classifier_original = FP32PyTorchClassifier(   
                model=model_original,
                clip_values=self._get_clip_values(),
                loss=F.cross_entropy,
                input_shape=self.args.input_shape[self.dataset_name],
                nb_classes=self.num_classes,
            )

            attack_original = HopSkipJump(
                classifier=classifier_original,
                targeted=False,
                max_iter=self.args.hs_max_iter,
                max_eval=self.args.hs_max_eval,
                batch_size=1,
            )
            limit = self.args.target_unlearning_size if name == "target" else self.args.shadow_unlearning_size
            processed = 0

            for unlearning_set_index, unlearning_indices in unlearning_set.items():
                if processed >= limit:
                    break
                processed += 1

                if n % 5 == 0:
                    print((sample_index, unlearning_set_index))
                n += 1

                save_name_unlearning = save_path + 'unlearning_S' + str(sample_index) + '_' + str(unlearning_set_index)
                model_unlearning = torch.load(save_name_unlearning, weights_only=False)
                model_unlearning = ForceFloat32(model_unlearning).float().eval()  

                model_unlearning.query_num = 0

                classifier_unlearning = FP32PyTorchClassifier(  
                    model=model_unlearning,
                    clip_values=self._get_clip_values(),
                    loss=F.cross_entropy,
                    input_shape=self.args.input_shape[self.dataset_name],
                    nb_classes=self.num_classes,
                )

                attack_unlearning = HopSkipJump(
                    classifier=classifier_unlearning,
                    targeted=False,
                    max_iter=self.args.hs_max_iter,
                    max_eval=self.args.hs_max_eval,
                    batch_size=1,
                )

                # -------- member sample (deleted record)
                test_pos_case, y_pos = self._generate_test_case(unlearning_indices)

                x_pos = test_pos_case.detach().cpu().numpy().astype(np.float32)
                pred_o = int(np.argmax(classifier_original.predict(x_pos), axis=1)[0])
                pred_u = int(np.argmax(classifier_unlearning.predict(x_pos), axis=1)[0])
                if pred_o != int(y_pos) or pred_u != int(y_pos):
                    continue

                # --- original model (member)
                start = time.time()
                adv_before_pos = _maybe_hsj(x_pos, y_pos, classifier_original, attack_original)
                elapsed = time.time() - start
                q_count_orig = model_original.query_num
                model_original.query_num = 0
                csv_writer.writerow([sample_index, unlearning_set_index, 1, "original", elapsed, q_count_orig])

                # --- unlearning model (member)
                start = time.time()
                adv_after_pos = _maybe_hsj(x_pos, y_pos, classifier_unlearning, attack_unlearning)
                elapsed = time.time() - start
                q_count_unlearn = model_unlearning.query_num
                model_unlearning.query_num = 0

                csv_writer.writerow([sample_index, unlearning_set_index, 1, "unlearning", elapsed, q_count_unlearn])

                # ذخیره در دیتافریم: همه چیز numpy float32 و هم‌مقیاس
                x_pos_store = x_pos.copy().astype(np.float32)              # (1,1,28,28)
                adv_before_pos = adv_before_pos.astype(np.float32)
                adv_after_pos  = adv_after_pos.astype(np.float32)

                for i in range(x_pos_store.shape[0]):
                    adv_ex_df.loc[len(adv_ex_df)] = [x_pos_store[i], adv_before_pos[i], adv_after_pos[i], 1]

                # -------- non-member sample
                local_seed = int(self.args.seed) * 1000003 + sample_index * 1009 + unlearning_set_index
                rng = np.random.default_rng(local_seed)
                neg_index = rng.choice(self.record_split.negative_indices, size=unlearning_indices.size, replace=False)

                test_neg_case, y_neg = self._generate_test_case(neg_index)
                x_neg = test_neg_case.detach().cpu().numpy().astype(np.float32)

                # فیلتر مثل member
                #pred_o = int(np.argmax(classifier_original.predict(x_neg), axis=1)[0])
                #pred_u = int(np.argmax(classifier_unlearning.predict(x_neg), axis=1)[0])
                #if pred_o != int(y_neg) or pred_u != int(y_neg):
                #    continue

                # --- original model (non-member)
                start = time.time()
                adv_before_neg = _maybe_hsj(x_neg, y_neg, classifier_original, attack_original)
                elapsed = time.time() - start
                q_count_orig_neg = model_original.query_num
                model_original.query_num = 0
                csv_writer.writerow([sample_index, unlearning_set_index, 0, "original", elapsed, q_count_orig_neg])

                # --- unlearning model (non-member)
                start = time.time()
                adv_after_neg = _maybe_hsj(x_neg, y_neg, classifier_unlearning, attack_unlearning)
                elapsed = time.time() - start
                q_count_unlearn_neg = model_unlearning.query_num
                model_unlearning.query_num = 0
                csv_writer.writerow([sample_index, unlearning_set_index, 0, "unlearning", elapsed, q_count_unlearn_neg])

                x_neg_store = x_neg.copy().astype(np.float32)
                adv_before_neg = adv_before_neg.astype(np.float32)
                adv_after_neg  = adv_after_neg.astype(np.float32)

                for i in range(x_neg_store.shape[0]):
                    adv_ex_df.loc[len(adv_ex_df)] = [x_neg_store[i], adv_before_neg[i], adv_after_neg[i], 0]

        csv_file.close()
        print(f"[{name}] adv_ex_df rows =", len(adv_ex_df))

        return adv_ex_df



class ActionAttackSisa(ActionAttack):
    def __init__(self, args):
        super(ActionAttackSisa, self).__init__(args)

        self.train_adv_ex_df = self.obtain_adversarial_example('shadow')  # HopSkipJump
        torch.save(self.train_adv_ex_df, os.path.join(self.attack_path, 'sisa_train_adv_ex_df'))
        self.test_adv_ex_df = self.obtain_adversarial_example('target')
        torch.save(self.test_adv_ex_df, os.path.join(self.attack_path, 'sisa_test_adv_ex_df'))
        self.train_adv_ex_df = torch.load(os.path.join(self.attack_path, 'sisa_train_adv_ex_df'))
        self.test_adv_ex_df = torch.load(os.path.join(self.attack_path, 'sisa_test_adv_ex_df'))
        self.train_dataset = torch.load(os.path.join(self.attack_path, 'attack_train_dataset'))
        self.test_dataset = torch.load(os.path.join(self.attack_path, 'attack_test_dataset'))

        self.launch_attack()

    def obtain_adversarial_example(self, name):
        n = 0
        self.record_split.generate_sample(name)
        adv_ex_df = pd.DataFrame(columns=['original', 'original model', 'unlearning model', 'mem or non-mem'])

        # --- CSV time logging setup
        csv_path = os.path.join(os.path.join(self.attack_path, f"time_log_{name}_sisa.csv"))
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["sample_index", "unlearning_set_index", "label", "model_type", "elapsed_time_sec", "query_count"])

        if name == 'target':
            save_path = config.TARGET_MODEL_PATH + self.save_name + '/'
            set_num = self.args.target_set_num
            num_shard = self.args.target_num_shard
        else:
            save_path = config.SHADOW_MODEL_PATH + self.save_name + '/'
            set_num = self.args.shadow_set_num
            num_shard = self.args.shadow_num_shard

        for sample_index in range(set_num):
            sample_set = self.record_split.sample_set[sample_index]
            unlearning_indices = sample_set["unlearning_indices"]
            unlearning_shard_mapping = sample_set["unlearning_shard_mapping"]

            original_models = []
            dim = {
                'mnist': (1, 10),
                'cifar10': (3, 10),
                'cifar100': (3, 100),
                'stl10': (3, 10),
            }

            original_models = []
            for shard_index in range(num_shard):
                shard_save_name = save_path + "original_S%s_M%s" % (sample_index, shard_index)
                original_models.append(torch.load(shard_save_name, weights_only=False))

                original_model = FusedModel(original_models, dim[self.args.dataset_name])
                original_model = ForceFloat32(original_model).float().eval()

                classifier_original = FP32PyTorchClassifier(
                model=original_model,
                clip_values=self._get_clip_values(),
                loss=F.cross_entropy,
                input_shape=self.args.input_shape[self.dataset_name],
                nb_classes=self.num_classes,
            )

            attack_original = HopSkipJump(
                classifier=classifier_original,
                targeted=False,
                max_iter=self.args.hs_max_iter,
                max_eval=self.args.hs_max_eval,
                batch_size=1,
            )

            for unlearning_set_index, unlearning_index in enumerate(unlearning_indices):
                if n % 5 == 0:
                    print((sample_index, unlearning_set_index))
                    torch.save(adv_ex_df, os.path.join(self.attack_path, f'sisa_{name}_adv_ex_df'))
                n += 1

                unlearning_shard_index = unlearning_shard_mapping[unlearning_index]
                prefix = "unlearning_S%s_M%s" % (sample_index, unlearning_shard_index)
                files = [f for f in os.listdir(save_path) if f.startswith(prefix)]

                if len(files) > 0:
                    shard_save_name = os.path.join(save_path, files[0])
                    shard_model = torch.load(shard_save_name, weights_only=False)
                else:
                    print(f"File not found with prefix: {prefix}")
                    continue

                unlearning_model = FusedModel(original_models, dim[self.args.dataset_name])
                unlearning_model.shard(unlearning_shard_index, shard_model)
                unlearning_model.eval()

                unlearning_model = ForceFloat32(unlearning_model).float().eval()

                classifier_unlearning = FP32PyTorchClassifier(
                    model=unlearning_model,
                    clip_values=self._get_clip_values(),
                    loss=F.cross_entropy,
                    input_shape=self.args.input_shape[self.dataset_name],
                    nb_classes=self.num_classes,
                )
                attack_unlearning = HopSkipJump(
                    classifier=classifier_unlearning,
                    targeted=False,
                    max_iter=self.args.hs_max_iter,
                    max_eval=self.args.hs_max_eval,
                    batch_size=1,
                )

                # -------- member sample
                test_pos_case, y_pos = self._generate_test_case(unlearning_index)

                start = time.time()
                x_pos = test_pos_case.detach().cpu().numpy().astype(np.float32)
                #pred_o = int(np.argmax(classifier_original.predict(x_pos), axis=1)[0])
                #pred_u = int(np.argmax(classifier_unlearning.predict(x_pos), axis=1)[0])
                #if pred_o != int(y_pos) or pred_u != int(y_pos):
                #   continue
                adv_before_pos = attack_original.generate(x=x_pos)

                elapsed = time.time() - start
                q_sisa = original_model.query_num
                original_model.query_num = 0
                csv_writer.writerow([sample_index, unlearning_set_index, 1, "original", elapsed, q_sisa])

                start = time.time()
                adv_after_pos  = attack_unlearning.generate(x=x_pos)

                elapsed = time.time() - start
                q_sisa_un = unlearning_model.query_num
                unlearning_model.query_num = 0
                csv_writer.writerow([sample_index, unlearning_set_index, 1, "unlearning", elapsed, q_sisa_un])

                x_pos = np.array(test_pos_case)
                for i in range(adv_before_pos.shape[0]):
                    adv_ex_df.loc[len(adv_ex_df)] = [x_pos[i], adv_before_pos[i], adv_after_pos[i], 1]

                # -------- non-member sample
                local_seed = int(self.args.seed) * 1000003 + sample_index * 1009 + unlearning_set_index
                rng = np.random.default_rng(local_seed)
                neg_indices = rng.choice(self.record_split.negative_indices, unlearning_indices.size, replace=False)

                test_neg_case, _ = self._generate_test_case(neg_indices)

                # --- original model (non-member)
                start = time.time()
                x_neg = test_neg_case.detach().cpu().numpy().astype(np.float32)
                adv_before_neg = attack_original.generate(x=x_neg)

                elapsed = time.time() - start

                q_sisa_neg = original_model.query_num
                original_model.query_num = 0
                csv_writer.writerow([sample_index, unlearning_set_index, 0, "original", elapsed, q_sisa_neg])

                # --- unlearning model (non-member)
                start = time.time()
                adv_after_neg  = attack_unlearning.generate(x=x_neg)

                elapsed = time.time() - start

                q_sisa_un_neg = unlearning_model.query_num
                unlearning_model.query_num = 0
                csv_writer.writerow([sample_index, unlearning_set_index, 0, "unlearning", elapsed, q_sisa_un_neg])

                x_neg = np.array(test_neg_case)
                for i in range(adv_before_neg.shape[0]):
                    adv_ex_df.loc[len(adv_ex_df)] = [x_neg[i], adv_before_neg[i], adv_after_neg[i], 0]

        csv_file.close()
        return adv_ex_df

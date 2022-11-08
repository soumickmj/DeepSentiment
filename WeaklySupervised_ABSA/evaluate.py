# The code is completely from the weakly supervised code https://github.com/teapot123/JASen
import os
import numpy as np
from nltk import word_tokenize
from sklearn.metrics import *
import argparse
import nltk
from nltk.corpus import stopwords
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split
#from model import CNN
#from model1 import CNN
#from model2 import CNN
from model3 import CNN
import torch.nn.functional as F
from collections import defaultdict
from math import sqrt
import pickle
# The code is completely from the weakly supervised code https://github.com/teapot123/JASen
def get_marginal_temb(vec_file):
    print(vec_file)
    f = open(vec_file, 'r')
    contents = f.readlines()[1:]
    t_emb = []
    for content in contents:
        content = content.strip()
        tokens = content.split(' ')
        word = tokens[0]
        vec = tokens[1:]
        vec = [float(ele) for ele in vec]
        t_emb.append(np.array(vec))
    return np.array(t_emb)
# The code is completely from the weakly supervised code https://github.com/teapot123/JASen
def get_temb_from_w(word_emb,topic2id):
    t_emb = [np.array(word_emb[t]) for t in topic2id]
    return np.array(t_emb)
# The code is completely from the weakly supervised code https://github.com/teapot123/JASen
def get_emb(vec_file):
    print(vec_file)
    f = open(vec_file,'r')
    contents = f.readlines()[1:]
    word_emb = {}
    vocabulary = {}
    vocabulary_inv = {}
    for idx, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        word = tokens[0]
        word = tokens[0]
        vec = tokens[1:]
        vec = [float(ele) for ele in vec]
        word_emb[word] = np.array(vec)
        vocabulary[word] = idx
        vocabulary_inv[idx] = word
    print(f"# of vocabulary {len(vocabulary)}")
    return word_emb, vocabulary, vocabulary_inv
# The code is completely from the weakly supervised code https://github.com/teapot123/JASen
def get_joint_temb(vec_file):
    print(vec_file)
    f = open(vec_file, 'r')
    contents = f.readlines()[1:]
    t_emb = []
    senti_topic = []
    aspect_topic = []
    for content in contents:
        content = content.strip()
        tokens = content.split(' ')
        word = tokens[0]
        senti = word.split('(')[1].split(',')[0]
        aspect = word.split(')')[0].split(',')[1]
        if senti not in senti_topic:
            senti_topic.append(senti)
        if aspect not in aspect_topic:
            aspect_topic.append(aspect)
        vec = tokens[1:]
        vec = [float(ele) for ele in vec]
        t_emb.append(np.array(vec))
    return np.array(t_emb), senti_topic, aspect_topic
# The code is completely from the weakly supervised code https://github.com/teapot123/JASen
def generate_batch(batch):
    label = torch.cat([entry[1].unsqueeze(0) for entry in batch])
    text = []
    for entry in batch:
        length = len(entry[0])
        tmp = F.pad(torch.tensor(entry[0]), (0,100-len(entry[0])), 'constant', 0).unsqueeze(0)
        text.append(tmp)
        for i in range(100):
            if tmp[0][i] >= len(vocabulary):
                print(tmp[i])
    gt1 = torch.from_numpy(np.array([entry[2] for entry in batch]))
    gt2 = torch.from_numpy(np.array([entry[3] for entry in batch]))
    
    text = torch.cat(text)

    return text, label, gt1, gt2

# The code is completely from the weakly supervised code https://github.com/teapot123/JASen
def train_func(sub_train_, model, mode, optimizer):
    # Train the model
    train_loss = 0
    train_acc = 0
    pseudo_aspect_train_acc = 0
    pseudo_senti_train_acc = 0
    aspect_train_acc = 0
    senti_train_acc = 0
    data = DataLoader(sub_train_, batch_size=batch_size, shuffle=True,
                      collate_fn=generate_batch)
    for text, cls, gt1, gt2 in data:
        # print(f'size of text: {text.size()}')
        optimizer.zero_grad()
        # text, cls, gt1, gt2 = text.to(device), cls.to(device), gt1.to(device), gt2.to(device)
        output = model(text)
        # loss = criterion(output, cls)
        loss = kl_criterion(torch.log(F.softmax(output, dim=-1)), cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if mode == 'joint':
            pseudo_aspect_train_acc += (output.argmax(1) % len(aspect_topic) == cls.argmax(1) % len(aspect_topic)).sum().item()
            pseudo_senti_train_acc += (output.argmax(1) / len(aspect_topic) == cls.argmax(1) / len(aspect_topic)).sum().item()
            aspect_train_acc += (output.argmax(1) % len(aspect_topic) == gt1).sum().item()
            senti_train_acc += (output.argmax(1) / len(aspect_topic) == (1-gt2)).sum().item()
        elif mode == 'aspect':
            pseudo_aspect_train_acc += (output.argmax(1) == cls.argmax(1)).sum().item()
            aspect_train_acc += (output.argmax(1) == gt1).sum().item()
        elif mode == 'senti':
            pseudo_senti_train_acc += (output.argmax(1) == cls.argmax(1)).sum().item()
            senti_train_acc += (output.argmax(1) == (1-gt2)).sum().item()


    # Adjust the learning rate
    # scheduler.step()
    if mode == 'joint':
        joint_scheduler.step()
    elif mode == 'aspect':
        aspect_scheduler.step()
    elif mode == 'senti':
        senti_scheduler.step()

    return train_loss / len(sub_train_), aspect_train_acc / len(sub_train_), senti_train_acc / len(sub_train_), pseudo_aspect_train_acc / len(sub_train_), pseudo_senti_train_acc / len(sub_train_)

# The code is completely from the weakly supervised code https://github.com/teapot123/JASen
def batch_train_func(text, target, model, aspect):

    # Train the model
    train_loss = 0
    train_acc = 0
    pseudo_aspect_train_acc = 0
    pseudo_senti_train_acc = 0
    aspect_train_acc = 0
    senti_train_acc = 0
    flag = True

    # print(f'size of text: {text.size()}')
    optimizer[aspect].zero_grad()
    # text, cls, gt1, gt2 = text.to(device), cls.to(device), gt1.to(device), gt2.to(device)
    output = model(text)
    loss = kl_criterion(torch.log(F.softmax(output, dim=-1)), target)
    # loss = criterion(output, cls)
    train_loss += loss.item()
    loss.backward()
    optimizer[aspect].step()

    # Adjust the learning rate
    scheduler[aspect].step()

    return 

# The code is completely from the weakly supervised code https://github.com/teapot123/JASen
def test(data_, model, mode):
    loss = 0
    acc = 0
    pseudo_aspect_test_acc = 0
    pseudo_senti_test_acc = 0
    aspect_test_acc = 0
    senti_test_acc = 0
    data = DataLoader(data_, batch_size=128, collate_fn=generate_batch)
    pred_distribution = []
    for text, cls, gt1, gt2 in data:
        # text, cls, gt1, gt2 = text.to(device), cls.to(device), gt1.to(device), gt2.to(device)
        with torch.no_grad():
            output = model(text)
            loss = kl_criterion(torch.log(F.softmax(output, dim=-1)), cls)

            if mode == 'joint':
                pseudo_aspect_test_acc += (output.argmax(1) % len(aspect_topic) == cls.argmax(1) % len(aspect_topic)).sum().item()
                pseudo_senti_test_acc += (output.argmax(1) / len(aspect_topic) == cls.argmax(1) / len(aspect_topic)).sum().item()
                aspect_test_acc += (output.argmax(1) % len(aspect_topic) == gt1).sum().item()
                senti_test_acc += (output.argmax(1) / len(aspect_topic) == (1-gt2)).sum().item()

            elif mode == 'aspect':
                pseudo_aspect_test_acc += (output.argmax(1) == cls.argmax(1) ).sum().item()
                aspect_test_acc += (output.argmax(1)  == gt1).sum().item()
            elif mode == 'senti':
                pseudo_senti_test_acc += (output.argmax(1) == cls.argmax(1) ).sum().item()
                senti_test_acc += (output.argmax(1)  == (1-gt2)).sum().item()

            pred_distribution.append(output)

    return loss / len(data_), aspect_test_acc / len(data_), senti_test_acc / len(data_), pseudo_aspect_test_acc / len(data_), pseudo_senti_test_acc / len(data_), torch.cat(pred_distribution, dim=0)

# The code is completely from the weakly supervised code https://github.com/teapot123/JASen
def print_info(model_name, train_loss, aspect_test_acc, senti_test_acc, pseudo_aspect_test_acc, pseudo_senti_test_acc):
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs %= 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'{model_name}\tLoss: {train_loss:.4f}(train)\t|\t')
    # print(f'Apsect Acc: {aspect_train_acc * 100:.1f}% ')
    # print(f'Senti Acc: {senti_train_acc * 100:.1f}% ')
    # print(f'Pseudo Apsect Acc: {pseudo_aspect_train_acc * 100:.1f}% ')
    # print(f'Pseudo Senti Acc: {pseudo_senti_train_acc * 100:.1f}% ')
    if (aspect_test_acc > 0):
        print(f'Apsect Acc: {aspect_test_acc * 100:.1f}% ')
    if (senti_test_acc > 0):
        print(f'Senti Acc: {senti_test_acc * 100:.1f}% ')
    if (pseudo_aspect_test_acc > 0):
        print(f'Pseudo Apsect Acc: {pseudo_aspect_test_acc * 100:.1f}% ')
    if (pseudo_senti_test_acc > 0):
        print(f'Pseudo Senti Acc: {pseudo_senti_test_acc * 100:.1f}% ')
    # print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')#

# The code is completely from the weakly supervised code https://github.com/teapot123/JASen
def target_score(logits, aspect):

    preds = torch.nn.Softmax(dim=-1)(logits)  # batch * class
    print(dataset)
    #if aspect == 'aspect' and dataset == 'datasets/laptop':
    if aspect == 'aspect' and dataset == 'datasets/restaurant':
        weight = preds**1.2 #/ torch.sum(preds, dim=0)
    else:
        weight = preds**2 / torch.sum(preds, dim=0)

    return (weight.t() / torch.sum(weight, dim=1)).t() 

# The code is completely from the weakly supervised code https://github.com/teapot123/JASen
def high_conf_train_loader(all_text, gt, preds, class_num, conf_threshold=0.8):
    
    valid_idx = []
    threshold_benchmark = [0.8, 0.85, 0.9, 0.93, 0.95, 0.97]
    threshold_idx = defaultdict(list)

    preds = F.softmax(preds)
    all_conf, all_pred_labels = torch.max(preds, dim=-1)

    valid_idx = all_conf > conf_threshold

    for threshold in threshold_benchmark:
        match_idx = all_conf > threshold
        print(f"Threshold: {threshold}; num_samples: {len(all_pred_labels[match_idx])}; acc: {torch.sum(all_pred_labels[match_idx] == gt[match_idx]).item() / len(gt[match_idx])}")

    current_text = all_text[valid_idx]
    labels = all_pred_labels[valid_idx]
    y_onehot = torch.FloatTensor(len(labels), class_num)
    y_onehot.zero_()
    y_onehot.scatter_(1, labels.unsqueeze(1), 1)

    return current_text, y_onehot

# The code is completely from the weakly supervised code https://github.com/teapot123/JASen
def reorder(target_scores, pred_labels, train_dataset, gt, all_text):
    all_conf, all_pred_labels = torch.max(target_scores, dim=-1)
    _, sorted_idx = torch.sort(all_conf, descending=True)
    target_scores = target_scores[sorted_idx]
    pred_labels = pred_labels[sorted_idx]
    gt = gt[sorted_idx]
    new_dataset = [train_dataset[i] for i in sorted_idx]
    all_text = all_text[sorted_idx]

    return target_scores, pred_labels, new_dataset, gt, all_text

# The code is completely from the weakly supervised code https://github.com/teapot123/JASen
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--dataset', default='datasets/laptop')
    parser.add_argument('--dataset', default='datasets/restaurant')
    parser.add_argument('--topic_file', default='mix')
    parser.add_argument('--aspect_file', default='aspect,senti')
    parser.add_argument('--test_file', default='test.txt')
    parser.add_argument('--xsepconv', nargs='*', type=int, default=[0, 0, 1])


    args = parser.parse_args()
    test_file = args.test_file
    dataset = args.dataset
    aspects = args.aspect_file.split(',')
    xsepconv = args.xsepconv

    text = []
    pred1 = []
    pred2 = []
    gt1 = []
    gt2 = []
    stop_words = set(stopwords.words('english'))

    with open(os.path.join(dataset, test_file)) as f:
        for line in f:
            tmp = line.split('\t')
            if len(tmp) != 4:
                continue
            gt1.append(int(tmp[1]))
            gt2.append(int(tmp[2]))
            s = ' '.join([w.lower() for w in word_tokenize(tmp[3].strip()) if w.lower() not in stop_words])
            text.append(s)

    #authors architecture rest
    #with open('./Pickledata/latest_rest_original_gt1.pkl','wb') as f:
    #doconv architecture rest
    #with open('./Pickledata/latest_rest_doconv_gt1.pkl', 'wb') as f:
    #xsepconv architecture rest
    #with open('./Pickledata/latest_rest_xsepconv_gt1.pkl', 'wb') as f:
    #2 xsepconv & 1 conv2d architecture rest
    #with open('./Pickledata/latest_rest_xsepconv2_gt1.pkl','wb') as f:
    # 1 xsepconv & 2 conv2d architecture lap
    #with open('./Pickledata/latest_rest_xsepconv3_gt1.pkl','wb') as f:
    #1 xsepconv & 2 conv2d architecture lap
    #with open('./Pickledata/latest_lap_xsepconv3_gt1.pkl','wb') as f:
    # 2 xsepconv & 1 conv2d architecture lap
    #with open('./Pickledata/latest_lap_xsepconv2_gt1.pkl', 'wb') as f:
    # 3 xsepconv architecture lap
    #with open('./Pickledata/latest_lap_xsepconv_gt1.pkl','wb') as f:
    # doconv architecture lap
    #with open('./Pickledata/latest_lap_doconv_gt1.pkl', 'wb') as f:
    # doconv architecture lap correct
    #with open('./Pickledata/latest_lap_doconv1_gt1.pkl','wb') as f:
    #authors lap
    #with open('./Pickledata/latest_lap_gt1.pkl','wb') as f:
    #      pickle.dump(gt1, f)
    # xsep dconv rest rq3
    # with open('./Pickledata/latest_lap_vd17da1_xsepdoconv_gt1.pkl','wb') as f:
    # doconv enh rest vdcn9
    #with open('./Pickledata/latest_rest_vd9_xsepdoconv_gt1.pkl','wb') as f:
    # doconv enh rest vdcnn9
    #with open('./Pickledata/latest_lap_vd9_xsepdoconv_gt1.pkl','wb') as f:
    # doconv enh lap vdcnn 17
    #with open('./Pickledata/latest_lap_vd17_xsepdoconv_gt1.pkl','wb') as f:
    # doconv enh rest vdcnn 17
    #with open('./Pickledata/latest_rest_vd17_xsepdoconv_gt1.pkl', 'wb') as f:
    # doconv enh rest vdcnn 29
    #with open('./Pickledata//latest_rest_vd29_xsepdoconv_gt1.pkl','wb') as f:
    # doconv enh rest vdcnn 29
    #with open('./Pickledata/latest_lap_vd29_xsepdoconv_gt1.pkl','wb') as f:
    # xsep dconv vd17 rest
    # with open('./Pickledata/latest_rest_vd17da_xsepdoconv_gt1.pkl','wb') as f:
    ###rq5
    #with open('./Pickledata/latest_lap_vd9_da_xsepdoconv_gt1.pkl','wb') as f:
    ###rq5 rest
    with open('./Pickledata/latest1_rest_vd9_da_xsepdoconv_gt1.pkl','wb') as f:
    #architecture ae rest
    #with open('./Pickledata/latest_rest_ae_gt1.pkl','wb') as f:
    #doconv
    #with open('./Pickledata/latest_rest_ae_doconv_gt1.pkl','wb') as f:
    #xsepconv
    #with open('./Pickledata/latest_rest_ae_xsep_gt1.pkl','wb') as f:
    #xsep rest ae xsep aspects
    #with open('./Pickledata/latest_re_ae_xsep_gt1.pkl','wb') as f:
    #doconv rest
    #with open('./Pickledata/latest_re_ae_doconv_gt1.pkl','wb') as f:
    #conv rest ae
    #with open('./Pickledata/latest_re_ae_conv_gt1.pkl','wb') as f:
    # xsep conv lap
    #with open('./Pickledata/latest_lap_ae_xsepconv_gt1.pkl','wb') as f:
    # xsep doconv lap
    #with open('./Pickledata/latest_lap_ae_xsepdoconv_gt1.pkl','wb') as f:
    # xsep xsep lap
    #with open('./Pickledata/latest_lap_ae_xsepxsep_gt1.pkl','wb') as f:
    #  doconv xsep rest ae
    #with open('./Pickledata/latest_lap_ae_xsep_doconv_gt1.pkl','wb') as f:
    # doconv lap ae doconv
    #with open('./Pickledata/latest_lap_ae_doconv_doconv_gt1.pkl', 'wb') as f:
    #conv lap ae conv
    #with open('./Pickledata/latest_lap_ae_conv_doconv_gt1.pkl', 'wb') as f:
    #doconv enh lap
    #with open('./Pickledata/latest_lap_ae_xsepdoconv1_gt1.pkl','wb') as f:
            pickle.dump(gt1, f)


    #authors architecture rest
    #with open('./Pickledata/latest_rest_original_gt2.pkl','wb') as f:
    #doconv architecture rest
    #with open('./Pickledata/latest_rest_doconv_gt2.pkl', 'wb') as f:
    #xsepconv architecture rest
    #with open('./Pickledata/latest_rest_xsepconv_gt2.pkl', 'wb') as f:
    #2 xsepconv & 1 conv2d architecture rest
    #with open('/home/sowmya/Documents/Thesis/CorrectCode/WeakABSA/Pickledata/latest_rest_xsepconv2_gt2.pkl','wb') as f:
    #      pickle.dump(gt2, f)
    #1 xsepconv & 2 conv2d architecture rest
    #with open('./Pickledata/latest_rest_xsepconv3_gt2.pkl', 'wb') as f:
    #1 xsepconv & 2 conv2d architecture lap
    #with open('./Pickledata/latest_lap_xsepconv3_gt2.pkl','wb') as f:
    # 1 xsepconv & 2 conv2d architecture lap
    #with open('./Pickledata/latest_lap_xsepconv2_gt2.pkl','wb') as f:
    # 3 xsepconv architecture lap
    #with open('./Pickledata/latest_lap_xsepconv_gt2.pkl','wb') as f:
    # doconv architecture lap
    #with open('./Pickledata/latest_lap_doconv_gt2.pkl', 'wb') as f:
    # doconv architecture lap
    #with open('./Pickledata/latest_lap_doconv1_gt2.pkl', 'wb') as f:
    # author architecture lap
    #with open('./Pickledata/latest_lap_gt2.pkl','wb') as f:
    #     pickle.dump(gt2, f)
    # xsep doconv ae enhanced rest rq3
    # with open('./Pickledata/latest_rest_vd17da1_xsepdoconv_gt2.pkl','wb') as f:
    # xsep doconv ae enhanced rest vd9
    #with open('./Pickledata/latest_rest_vd9_xsepdoconv_gt2.pkl','wb') as f:
    # xsep doconv ae enhanced lap vd9
    #with open('./Pickledata/latest_lap_vd9_xsepdoconv_gt2.pkl','wb') as f:
    # xsep doconv ae enhanced lap vd17
    #with open('./Pickledata/latest_lap_vd17_xsepdoconv_gt2.pkl','wb') as f:
    # xsep doconv ae enhanced rest vd17
    #with open('./Pickledata/latest_rest_vd17_xsepdoconv_gt2.pkl', 'wb') as f:
    # xsep doconv ae enhanced rest vd29
    #with open('./Pickledata/latest_rest_vd29_xsepdoconv_gt2.pkl','wb') as f:
    # xsep doconv ae enhanced lap vd29
    #with open('./Pickledata/latest_lap_vd29_xsepdoconv_gt2.pkl','wb') as f:
    # xsep doconv ae enhanced lap vd17
    # with open('./Pickledata/latest_rest_vd17_xsepdoconv_gt2.pkl', 'wb') as f:
    # xsep doconv ae enhanced rest rq5
    #with open('./Pickledata/latest_rest_vd9_da_xsepdoconv_gt2.pkl','wb') as f:
    # xsep doconv ae enhanced rest rq5
    with open('./Pickledata/latest_resttt_vd9_da_xsepdoconv_gt12.pkl','wb') as f:

    #architecture ae rest
    #with open('./Pickledata/latest_rest_ae_gt2.pkl', 'wb') as f:
    #doconv
    #with open('./Pickledata/latest_rest_ae_doconv_gt2.pkl','wb') as f:
    #xsep
    #with open('./Pickledata/latest_rest_ae_xsep_gt2.pkl','wb') as f:
    #doconv rest ae
    #with open('./Pickledata/latest_re_ae_xsep_gt2.pkl','wb') as f:
    #doconv ae rest doconv
    #with open('./Pickledata/latest_re_ae_doconv_gt2.pkl','wb') as f:
    #conv doconv ae
    #with open('./Pickledata/latest_re_ae_conv_doconv_gt2.pkl', 'wb') as f:
    # xsep conv ae
    #with open('./Pickledata/latest_lap_ae_xsepconv_gt2.pkl','wb') as f:
    # xsep doconv ae
    #with open('./Pickledata/latest_lap_ae_xsepdoconv_gt2.pkl','wb') as f:
    # xsep xsep lap
    #with open('./Pickledata/latest_lap_ae_xsepxsep_gt2.pkl','wb') as f:
    # xsep doconv ae
    #with open('./Pickledata/latest_lap_ae_xsep_doconv_gt2.pkl','wb') as f:
    #doconv xsep ae lap
    #with open('./Pickledata/latest_lap_ae_doconv_doconv_gt2.pkl','wb') as f:
    #conv xsep ae lap
    #with open('./Pickledata/latest_lap_ae_conv_doconv_gt2.pkl','wb') as f:
    # xsep doconv ae enhanced
    #with open('./Pickledata/latest_lap_ae_xsepdoconv1_gt2.pkl','wb') as f:
    # xsep doconv ae enhanced rest
    #with open('./Pickledata/latest_rest_ae_xsepdoconv1_gt2.pkl','wb') as f:
           pickle.dump(gt2, f)


    topic2id = {}
    id2topic = {}

    aspect = args.topic_file
    w_emb_file = 'emb_'+aspect + '_w.txt'
    t_emb_file = 'emb_'+aspect + '_t.txt'

    joint_word_emb, vocabulary, vocabulary_inv = get_emb(os.path.join(args.dataset, w_emb_file))
    joint_topic_emb, senti_topic, aspect_topic = get_joint_temb(vec_file=os.path.join(args.dataset, t_emb_file))
 
    marginal_w_emb = {}
    marginal_topic_emb = {}

    for aspect in aspects:
        w_emb_file = 'emb_'+aspect + '_w_kw_w.txt'
        t_emb_file = 'emb_'+aspect + '_w_kw_t.txt'


        marginal_w_emb[aspect], vocabulary1, _ = get_emb(os.path.join(args.dataset, w_emb_file))
        marginal_topic_emb[aspect] = get_marginal_temb(vec_file=os.path.join(args.dataset, t_emb_file))


    not_in_vocab = 0
    zero_sen = 0
    joint_old_pred = []
    aspect_old_pred = []
    senti_old_pred = []
    old_pred = []
    old_pred_aspect = []
    old_pred_senti = []
    temperature = 20.0
    

    for k,s in enumerate(text):
        tmp = np.sum([1 if w in vocabulary else 0 for w in s.split(' ')])
        not_in_vocab += len(s.split(' ')) - tmp
        if tmp == 0:
            zero_sen += 1
            print(s)
        s_rep = np.sum([joint_word_emb[w] if w in vocabulary else np.zeros((100)) for w in s.split(' ')], axis=0)/len(text)
        s_norm = np.linalg.norm(s_rep)
        if s_norm > 0:
            joint_dot = np.dot(s_rep, np.transpose(joint_topic_emb))/np.linalg.norm(joint_topic_emb, axis=1)/s_norm
        else:
            joint_dot = np.dot(s_rep, np.transpose(joint_topic_emb))/np.linalg.norm(joint_topic_emb, axis=1)
        joint_old_pred.append(F.softmax(temperature * torch.tensor(joint_dot).float()))
        # joint_sum = np.sum(joint_dot)
        # joint_dist = joint_dot / joint_sum

        joint_dot_sum = {}
        joint_dot_sum['aspect'] = joint_dot
        joint_dot_sum['senti'] = joint_dot
        

        for i, aspect in enumerate(aspects):
            s_rep = np.sum([marginal_w_emb[aspect][w] if w in vocabulary1 else np.zeros((100)) for w in s.split(' ')], axis=0)/len(text)
            s_norm = np.linalg.norm(s_rep)
            if s_norm > 0:
                marginal_dot = np.dot(s_rep, np.transpose(marginal_topic_emb[aspect]))/np.linalg.norm(marginal_topic_emb[aspect], axis=1)/s_norm
            else:
                marginal_dot = np.dot(s_rep, np.transpose(marginal_topic_emb[aspect]))/np.linalg.norm(marginal_topic_emb[aspect], axis=1)
            if i == 0:
                aspect_old_pred.append(F.softmax(temperature * torch.tensor(marginal_dot).float()))
                for j in range(len(joint_dot)):
                    joint_dot_sum[aspect][j] += marginal_dot[j % len(marginal_topic_emb['aspect'])]
            else:
                senti_old_pred.append(F.softmax(temperature * torch.tensor(marginal_dot).float()))
                for j in range(len(joint_dot)):
                    joint_dot_sum[aspect][j] += marginal_dot[int(j/len(marginal_topic_emb['aspect']))]
        

        old_pred.append(temperature * torch.tensor(joint_dot).float())
        joint_dot_sum['aspect'] = joint_dot_sum['aspect'].reshape((2,len(marginal_topic_emb['aspect'])))
        joint_dot_sum['senti'] = joint_dot_sum['senti'].reshape((2,len(marginal_topic_emb['aspect'])))
        old_pred_aspect.append(temperature * torch.tensor(joint_dot_sum['aspect']).float())
        old_pred_senti.append(temperature * torch.tensor(joint_dot_sum['senti']).float())
        
        label2 = np.argmax(np.sum(joint_dot_sum['senti'], axis=1))
        label1 = np.argmax(np.sum(joint_dot_sum['aspect'], axis=0))
        pred1.append(label1)
        pred2.append(1-label2)
        # fout.write(aspect_topic[gt1[k]]+' '+aspect_topic[label1]+' '+senti_topic[1-gt2[k]]+' '+senti_topic[label2]+'\n')

    acc = accuracy_score(gt1, pred1)
    p = precision_score(gt1, pred1, average='macro')
    r = recall_score(gt1, pred1, average='macro')
    f1_mac = f1_score(gt1, pred1, average='macro')
    
    print(f"Aspect Accuracy: {acc} Precision: {p} Recall: {r} mac-F1: {f1_mac}")
    print(confusion_matrix(gt1, pred1))
    print(f"not in vocab: {not_in_vocab}  zero sentence: {zero_sen}")

    acc = accuracy_score(gt2, pred2)
    p = precision_score(gt2, pred2, average='macro')
    r = recall_score(gt2, pred2, average='macro')
    f1_mac = f1_score(gt2, pred2, average='macro')
    
    print(f"Sentiment Accuracy: {acc} Precision: {p} Recall: {r} mac-F1: {f1_mac}")
    print(confusion_matrix(gt2, pred2))
    print(f"not in vocab: {not_in_vocab}  zero sentence: {zero_sen}")



    print("start training CNN\n")

    learning_rate = 0.05
    batch_size = 16
    output_size = len(senti_topic) * len(aspect_topic)
    embedding_length = 100
    N_EPOCHS_PRE = 7
    N_EPOCHS = 200
    self_training = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    min_valid_loss = float('inf')

    train_len = len(text)
    train_dataset_joint = []
    train_dataset_aspect = []
    train_dataset_senti = []
    for i, t in enumerate(text):
        s_index = [vocabulary[w] if w in vocabulary else 0 for w in t.split(' ')]
        # train_dataset_joint.append([s_index, joint_old_pred[i], gt1[i], gt2[i]])
        # train_dataset_aspect.append([s_index, aspect_old_pred[i], gt1[i], gt2[i]])
        # train_dataset_senti.append([s_index, senti_old_pred[i], gt1[i], gt2[i]])
        train_dataset_joint.append([s_index, F.softmax(old_pred[i]), gt1[i], gt2[i]])
        train_dataset_aspect.append([s_index, F.softmax(torch.sum(old_pred_aspect[i].view(2,len(marginal_topic_emb['aspect'])), dim=0)/2), gt1[i], gt2[i]])
        train_dataset_senti.append([s_index, F.softmax(torch.sum(old_pred_senti[i].view(2,len(marginal_topic_emb['aspect'])), dim=1)/len(marginal_topic_emb['aspect'])), gt1[i], gt2[i]])    
    # print(len(train_dataset))

    joint_embedding = torch.zeros((len(vocabulary)+1, 100))
    aspect_embedding = torch.zeros((len(vocabulary)+1, 100))
    senti_embedding = torch.zeros((len(vocabulary)+1, 100))
    for i in vocabulary_inv:
        joint_embedding[i+1] = torch.tensor(joint_word_emb[vocabulary_inv[i]])
        aspect_embedding[i+1] = torch.tensor(marginal_w_emb['aspect'][vocabulary_inv[i]])
        senti_embedding[i+1] = torch.tensor(marginal_w_emb['senti'][vocabulary_inv[i]])

    #code modified based on architecture experiments primarily based on the weakly supervised code https://github.com/teapot123/JASen
    # topic_repeat = torch.from_numpy(topic_emb).repeat(1,3).float()
    #authors & doconv architecture
    # joint_model = CNN(batch_size, output_size, 1, 20, [2,3,4], 1, 0, 0.0, len(vocabulary)+1, 100, joint_embedding)
    # aspect_model = CNN(batch_size, len(aspect_topic), 1, 20, [2,3,4], 1, 0, 0.0, len(vocabulary)+1, 100, aspect_embedding)
    # senti_model = CNN(batch_size, len(senti_topic), 1, 20, [2,3,4], 1, 0, 0.0, len(vocabulary)+1, 100, senti_embedding)
    #xsepconv
    # joint_model = CNN(batch_size, output_size, 1, 20, [2,3,4], 1, 0, 0.0, len(vocabulary)+1, 100, joint_embedding,xsepconv)
    # aspect_model = CNN(batch_size, len(aspect_topic), 1, 20, [2,3,4], 1, 0, 0.0, len(vocabulary)+1, 100, aspect_embedding,xsepconv)
    # senti_model = CNN(batch_size, len(senti_topic), 1, 20, [2,3,4], 1, 0, 0.0, len(vocabulary)+1, 100, senti_embedding,xsepconv)
    #  doconv vdcnn 9 architecture
    joint_model = CNN(batch_size, output_size, 1, 20, [2, 3, 4, 3, 3, 3, 3, 3, 3], 1, 0, 0.0, len(vocabulary) + 1, 100, joint_embedding)
    aspect_model = CNN(batch_size, len(aspect_topic), 1, 20, [2, 3, 4, 3, 3, 3, 3, 3, 3], 1, 0, 0.0, len(vocabulary) + 1, 100, aspect_embedding)
    senti_model = CNN(batch_size, len(senti_topic), 1, 20, [2, 3, 4, 3, 3, 3, 3, 3, 3], 1, 0, 0.0, len(vocabulary) + 1, 100,senti_embedding)
    #  doconv vdcnn 17 architecture
    # joint_model = CNN(batch_size, output_size, 1, 20, [2, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 1, 0, 0.0, len(vocabulary) + 1, 100, joint_embedding)
    # aspect_model = CNN(batch_size, len(aspect_topic), 1, 20, [2, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 1, 0, 0.0, len(vocabulary) + 1, 100, aspect_embedding)
    # senti_model = CNN(batch_size, len(senti_topic), 1, 20, [2, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 1, 0, 0.0, len(vocabulary) + 1, 100,senti_embedding)
    #  doconv vdcnn 29 architecture
    # joint_model = CNN(batch_size, output_size, 1, 20, [2, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 1, 0, 0.0, len(vocabulary) + 1, 100, joint_embedding)
    # aspect_model = CNN(batch_size, len(aspect_topic), 1, 20, [2, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 1, 0, 0.0, len(vocabulary) + 1, 100, aspect_embedding)
    # senti_model = CNN(batch_size, len(senti_topic), 1, 20, [2, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 1, 0, 0.0, len(vocabulary) + 1, 100, senti_embedding)

    criterion = torch.nn.CrossEntropyLoss()#.to(device)
    kl_criterion = torch.nn.KLDivLoss()
    joint_model_optimizer = torch.optim.SGD(joint_model.parameters(), lr=learning_rate)
    joint_scheduler = torch.optim.lr_scheduler.StepLR(joint_model_optimizer, 1, gamma=0.9)
    aspect_model_optimizer = torch.optim.SGD(aspect_model.parameters(), lr=learning_rate)
    aspect_scheduler = torch.optim.lr_scheduler.StepLR(aspect_model_optimizer, 1, gamma=0.9)
    senti_model_optimizer = torch.optim.SGD(senti_model.parameters(), lr=learning_rate)
    senti_scheduler = torch.optim.lr_scheduler.StepLR(senti_model_optimizer, 1, gamma=0.9)
    
    # model.to(device)

    orig_text = [train[0] for train in train_dataset_joint]
    all_text = []
    all_text2 = []
    for entry in orig_text:
        length = len(entry)
        tmp = F.pad(torch.tensor(entry), (0,100-len(entry)), 'constant', 0).unsqueeze(0)
        all_text.append(tmp)
        all_text2.append(tmp)
        for i in range(100):
            if tmp[0][i] >= len(vocabulary):
                print(tmp[i])

    all_text = torch.cat(all_text)
    all_text2 = torch.cat(all_text2)

    for epoch in range(N_EPOCHS_PRE):

        start_time = time.time()
        # train_loss, aspect_train_acc, senti_train_acc, pseudo_aspect_train_acc, pseudo_senti_train_acc = train_func(train_dataset_joint, joint_model, 'joint', joint_model_optimizer)
        # valid_loss, aspect_test_acc, senti_test_acc, pseudo_aspect_test_acc, pseudo_senti_test_acc, joint_pred_distribution = test(train_dataset_joint, joint_model, 'joint')
        # print_info('Joint Model', train_loss, aspect_test_acc, senti_test_acc, pseudo_aspect_test_acc, pseudo_senti_test_acc)
        
        train_loss, aspect_train_acc, _, pseudo_aspect_train_acc, _ = train_func(train_dataset_aspect, aspect_model, 'aspect', aspect_model_optimizer)
        valid_loss, aspect_test_acc, _, pseudo_aspect_test_acc, _, aspect_pred_distribution = test(train_dataset_aspect, aspect_model, 'aspect')
        print_info('Aspect Model', train_loss, aspect_test_acc, 0, pseudo_aspect_test_acc, 0)
        
        train_loss, _, senti_train_acc, _, pseudo_senti_train_acc = train_func(train_dataset_senti, senti_model, 'senti', senti_model_optimizer)
        valid_loss, _, senti_test_acc, _, pseudo_senti_test_acc, senti_pred_distribution = test(train_dataset_senti, senti_model, 'senti')
        print_info('Sentiment Model', train_loss, 0, senti_test_acc, 0, pseudo_senti_test_acc)



    print("###### Self-Training #####")
    last_aspect_pred_labels = aspect_pred_distribution.argmax(-1)
    last_senti_pred_labels = senti_pred_distribution.argmax(-1)
    self_learning_rate = 0.001
    optimizer = {}
    scheduler = {}
    optimizer['aspect'] = torch.optim.SGD(aspect_model.parameters(), lr=self_learning_rate)
    optimizer['senti'] = torch.optim.SGD(senti_model.parameters(), lr=self_learning_rate)
    for aspect in aspects:
        scheduler[aspect] = torch.optim.lr_scheduler.StepLR(optimizer[aspect], int(len(train_dataset_aspect)/batch_size), gamma=0.9)
    cur_idx = 0
    update_interval = 60
    total_steps = len(train_dataset_aspect) * N_EPOCHS
    stop_aspect = False
    stop_senti = False

    start_time = time.time()

    for i in range(total_steps):
        if i % update_interval == 0:

            # evaluate current performance
            valid_loss, aspect_test_acc, _, pseudo_aspect_test_acc, _, aspect_pred_distribution = test(train_dataset_aspect, aspect_model, 'aspect')
            # print_info('Aspect Model', train_loss, aspect_test_acc, 0, pseudo_aspect_test_acc, 0)
            valid_loss, _, senti_test_acc, _, pseudo_senti_test_acc, senti_pred_distribution = test(train_dataset_senti, senti_model, 'senti')
            # print_info('Sentiment Model', train_loss, 0, senti_test_acc, 0, pseudo_senti_test_acc)

            aspect_target_scores = target_score(aspect_pred_distribution, 'aspect')
            aspect_pred_labels = aspect_pred_distribution.argmax(-1)
            senti_target_scores = target_score(senti_pred_distribution, 'senti')
            senti_pred_labels = senti_pred_distribution.argmax(-1)


            if i == 0:
                old_gt1, old_gt2, = gt1, gt2
                #authors architecture rest
                #with open('./Pickledata/latest_rest_original_old_gt1.pkl','wb') as f:
                #doconv architecture rest
                #with open('./Pickledata/latest_rest_doconv_old_gt1.pkl','wb') as f:
                #xsepconv architecture rest
                #with open('./Pickledata/latest_rest_xsepconv_old_gt1.pkl','wb') as f:
                #2 xsepconv & 2 conv architecture rest
                #with open('./Pickledata/latest_rest_xsepconv2_old_gt1.pkl','wb') as f:
                #     pickle.dump(old_gt1, f)
                # 1 xsepconv & 2 conv architecture rest
                #with open('./Pickledata/latest_rest_xsepconv3_old_gt1.pkl','wb') as f:
                # 1 xsepconv & 2 conv architecture lap
                #with open('./Pickledata/latest_lap_xsepconv3_old_gt1.pkl','wb') as f:
                # 2 xsepconv & 1 conv architecture lap
                #with open('./Pickledata/latest_lap_xsepconv2_old_gt1.pkl','wb') as f:
                #3 xsep
                #with open('./Pickledata/latest_lap_xsepconv_old_gt1.pkl','wb') as f:
                # doconv architecture lap
                #with open('./Pickledata/latest_lap_doconv_old_gt1.pkl','wb') as f:
                # xsep doconv rest rq3
                #with open('./Pickledata/latest_lap_vd17da1_xsep_doconv_old_gt1.pkl','wb') as f:
                # xsep rest doconv vd9
                #with open('./Pickledata/latest_rest_vd9_xsep_doconv_old_gt1.pkl','wb') as f:
                # xsep lap doconv vd9
                #with open('./Pickledata/latest_lap_vd9_xsep_doconv_old_gt1.pkl','wb') as f:
                # xsep lap doconv vd17
                #with open('./Pickledata/latest_lap_vd17_xsep_doconv_old_gt1.pkl','wb') as f:
                # xsep rest doconv vd17
                #with open('./Pickledata/latest_rest_vd17_xsep_doconv_old_gt1.pkl','wb') as f:
                # xsep rest doconv vd29
                #with open('./Pickledata/latest_rest_vd29_xsep_doconv_old_gt1.pkl','wb') as f:
                # xsep rest doconv vd9
                #with open('./Pickledata/latest_lap_vd29_xsep_doconv_old_gt1.pkl','wb') as f:
                # xsep doconv rest vd17
                # with open('./Pickledata/latest_rest_vd17da_xsep_doconv_old_gt1.pkl','wb') as f:
                # xsep doconv rest rq5
                #with open('./Pickledata/latest_lap_vd9_da_xsep_doconv_old_gt1.pkl','wb') as f:
                # xsep doconv rest rq5
                #with open('./Pickledata/latest_rest_vd9_da_xsep_doconv_old_gt1.pkl','wb') as f:
                # doconv architecture lap correct
                #with open( './Pickledata/latest_rest_ae_doconv1_old_gt1.pkl','wb') as f:
                #authors lap
                #with open('./Pickledata/latest_lap_old_gt1.pkl', 'wb') as f:
                # xsep architecture rest
                #with open('./Pickledata/latest_rest_ae_old_gt1.pkl','wb') as f:
                #doconv
                #with open( './Pickledata/latest_rest_ae_doconv_old_gt1.pkl','wb') as f:

                #xsep
                #with open('./Pickledata/latest_rest_ae_xsep_old_gt1.pkl','wb') as f:
                #xsep rest ae xsep
                #with open('./Pickledata/latest_re_ae_xsep_old_gt1.pkl','wb') as f:
                # doconv rest
                #with open('./Pickledata/latest_re_ae_doconv_old_gt1.pkl', 'wb') as f:
                #conv rest
                #with open('./Pickledata/latest_re_ae_conv_old_gt1.pkl', 'wb') as f:
                # xsep conv
                #with open('./Pickledata/latest_lap_ae_xsepconv_old_gt1.pkl','wb') as f:
                # xsep doconv
                #with open('./Pickledata/latest_lap_ae_xsepdoconv_old_gt1.pkl','wb') as f:
                # xsep xsep lap
                #with open('./Pickledata/latest_lap_ae_xsepxsep_old_gt1.pkl','wb') as f:
                # xsep lap doconv
                #with open('./Pickledata/latest_lap_ae_xsep_doconv_old_gt1.pkl','wb') as f:
                # doconv rest doconv
                #with open('./Pickledata/latest_lap_ae_doconv_doconv_old_gt1.pkl', 'wb') as f:
                #conv lap doconv
                #with open('./Pickledata/latest_lap_ae_conv_doconv_old_gt1.pkl','wb') as f:
                # xsep lap doconv
                #with open('./Pickledata/latest_lap_ae_xsep_doconv1_old_gt1.pkl','wb') as f:
                # xsep lap doconv
                #with open('./Pickledata/latest_rest_ae_xsep_doconv1_old_gt1.pkl','wb') as f:
                       #pickle.dump(old_gt1, f)

                #authors architecture rest
                #with open('./Pickledata/latest_rest_original_old_gt2.pkl','wb') as f:
                #doconv architecture rest
                #with open('./Pickledata/latest_rest_doconv_old_gt2.pkl','wb') as f:
                #xsepconv architecture rest
                #with open('./Pickledata/latest_rest_xsepconv_old_gt2.pkl','wb') as f:
                #2 xsepconv & 1 conv architecture rest
                #with open('./Pickledata/latest_rest_xsepconv2_old_gt2.pkl','wb') as f:
                # 1 xsepconv & 2 conv architecture rest
                #with open('./Pickledata/latest_rest_xsepconv3_old_gt2.pkl','wb') as f:
                # 1 xsepconv & 2 conv architecture lap
                #with open('./Pickledata/latest_lap_xsepconv3_old_gt2.pkl','wb') as f:
                # 2 xsepconv & 1 conv architecture lap
                #with open('./Pickledata/latest_lap_xsepconv2_old_gt2.pkl','wb') as f:
                # 3 xsepconv lap
                #with open('./Pickledata/latest_lap_xsepconv_old_gt2.pkl','wb') as f:
                # doconv architecture rest
                #with open('./Pickledata/latest_lap_doconv_old_gt2.pkl','wb') as f:
                # doconv architecture rest correct
                #with open('./Pickledata/latest_lap_doconv1_old_gt2.pkl', 'wb') as f:
                # author architecture lap
                #with open('./Pickledata/latest_lap_old_gt2.pkl','wb') as f:
                # xsep rest doconv rq3
                #with open('./Pickledata/latest_lap_vd17da1_xsep_doconv_old_gt2.pkl','wb') as f:
                # xsep rest doconv vd9
                #with open('./Pickledata/latest_rest_vd9_xsep_doconv_old_gt2.pkl','wb') as f:
                # xsep rest doconv vd9
                #with open('./Pickledata/latest_lap_vd9_xsep_doconv_old_gt2.pkl','wb') as f:
                # xsep rest doconv vd17
                #with open('./Pickledata/latest_lap_vd17_xsep_doconv_old_gt2.pkl','wb') as f:
                # xsep rest doconv vd17
                #with open('./Pickledata/latest_rest_vd17_xsep_doconv_old_gt2.pkl','wb') as f:
                # xsep rest doconv vd29
                #with open('./Pickledata/latest_rest_vd29_xsep_doconv_old_gt2.pkl','wb') as f:
                # xsep rest doconv vd29
                #with open('./Pickledata/latest_lap_vd29_xsep_doconv_old_gt2.pkl','wb') as f:
                # xsep rest doconv vd17 da
                # with open('./Pickledata/latest_rest_vd17da_xsep_doconv_old_gt2.pkl','wb') as f:
                # xsep rest doconv rq5
                #with open('./Pickledata/latest_lap_vd9_da_xsep_doconv_old_gt2.pkl','wb') as f:
                # xsep rest doconv rq5
                #with open('./Pickledata/latest_rest_vd9_da_xsep_doconv_old_gt2.pkl','wb') as f:
                #xsep architecture author
                #with open('./Pickledata/latest_rest_ae_old_gt2.pkl','wb') as f:
                #doconvsep
                #with open('./Pickledata/latest_rest_ae_doconv_old_gt2.pkl','wb') as f:
                #xsep
                #with open('./Pickledata/latest_rest_ae_xsep_old_gt2.pkl','wb') as f:
                #xsep rest
                #with open('./Pickledata/latest_re_ae_xsep_old_gt2.pkl','wb') as f:
                # doconv rest
                #with open('./Pickledata/latest_re_ae_doconv_old_gt2.pkl','wb') as f:
                #conv rest
                #with open('./Pickledata/latest_re_ae_conv_old_gt2.pkl','wb') as f:
                # xsep lap doconv
                #with open('./Pickledata/latest_lap_ae_xsepconv_old_gt2.pkl','wb') as f:
                # xsep lap doconv
                #with open('./Pickledata/latest_lap_ae_xsepdoconv_old_gt2.pkl','wb') as f:
                # xsep lap xsep
                #with open('./Pickledata/latest_lap_ae_xsepxsep_old_gt2.pkl','wb') as f:
                # xsep rest doconv
                #with open('./Pickledata/latest_lap_ae_xsep_doconv_old_gt2.pkl','wb') as f:
                # do conv rest doconv
                #with open('./Pickledata/latest_lap_ae_doconv_doconv_old_gt2.pkl','wb') as f:
                # conv rest doconv
                #with open('./Pickledata/latest_lap_ae_conv_doconv_old_gt2.pkl','wb') as f:
                # xsep lap doconv
                #with open('./Pickledata/latest_lap_ae_xsep_doconv1_old_gt2.pkl','wb') as f:
                # xsep rest doconv
                #with open('./Pickledata/latest_rest_ae_xsep_doconv1_old_gt2.pkl','wb') as f:
                        #pickle.dump(old_gt2, f)

                old_train_dataset_aspect, old_train_dataset_senti = train_dataset_aspect, train_dataset_senti
                aspect_target_scores, aspect_pred_labels, train_dataset_aspect, gt1, all_text = reorder(aspect_target_scores, aspect_pred_labels, train_dataset_aspect, torch.tensor(gt1), all_text)
                senti_target_scores, senti_pred_labels, train_dataset_senti, gt2, all_text2 = reorder(senti_target_scores, senti_pred_labels, train_dataset_senti, torch.tensor(gt2), all_text2)
                last_aspect_pred_labels = aspect_pred_labels
                last_senti_pred_labels = senti_pred_labels

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60
            print('Step: %d' %(i + 1), " | time in %d minutes, %d seconds" %(mins, secs))


            print(f"Learning rate: {optimizer['aspect'].param_groups[0]['lr']:.4g}")

            p = precision_score(np.array(gt1), aspect_pred_labels.numpy(), average='macro')
            r = recall_score(np.array(gt1), aspect_pred_labels.numpy(), average='macro')
            f1_mac = f1_score(np.array(gt1), aspect_pred_labels.numpy(), average='macro')
            print(f"Aspect Acc: {aspect_test_acc * 100:.2f}% P: {p * 100:.2f}% R: {r * 100:.2f}% mac-F1: {f1_mac * 100:.2f}%")
            
            p = precision_score(1-np.array(gt2), senti_pred_labels.numpy(), average='macro')
            r = recall_score(1-np.array(gt2), senti_pred_labels.numpy(), average='macro')
            f1_mac = f1_score(1-np.array(gt2), senti_pred_labels.numpy(), average='macro')
            print(f"Senti Acc: {senti_test_acc * 100:.2f}% P: {p * 100:.2f}% R: {r * 100:.2f}% mac-F1: {f1_mac * 100:.2f}%")
            # print(f'Pseudo Apsect Acc: {pseudo_aspect_test_acc * 100:.1f}% ')
            # print(f'Pseudo Senti Acc: {pseudo_senti_test_acc * 100:.1f}% ')
            
            aspect_delta_label = (1 - torch.sum(aspect_pred_labels == last_aspect_pred_labels).item() / len(aspect_pred_labels)) * 100
            print(f"Aspect Delta label: {(1 - torch.sum(aspect_pred_labels == last_aspect_pred_labels).item() / len(aspect_pred_labels))*100:.2f}%")
            last_aspect_pred_labels = aspect_pred_labels

            senti_delta_label = (1 - torch.sum(senti_pred_labels == last_senti_pred_labels).item() / len(senti_pred_labels))*100
            print(f"Senti Delta label: {(1 - torch.sum(senti_pred_labels == last_senti_pred_labels).item() / len(senti_pred_labels))*100:.2f}%")
            last_senti_pred_labels = senti_pred_labels

            cur_idx = 0
            
        if cur_idx + batch_size < len(aspect_target_scores):
            select_idx = list(range(cur_idx, cur_idx + batch_size))
        else:
            select_idx = list(range(cur_idx, len(aspect_target_scores))) + list(range((cur_idx + batch_size) % len(aspect_target_scores)))
        cur_idx = (cur_idx + batch_size) % len(aspect_target_scores)
        
        text = all_text[select_idx]
        aspect_target = aspect_target_scores[select_idx]
        text2 = all_text2[select_idx]
        senti_target = senti_target_scores[select_idx]
        
        if aspect_delta_label >= 0.01 or i < update_interval:
            batch_train_func(text, aspect_target, aspect_model, 'aspect')
        else:
            stop_aspect = True
        if senti_delta_label >= 0.01 or i < update_interval:
            batch_train_func(text2, senti_target, senti_model, 'senti')
        else:
            stop_senti = True
        if aspect_delta_label < 0.01 and senti_delta_label < 0.01 and i >= update_interval:
            #authors architecture rest
            #torch.save(aspect_model, './weights/latest_rest_original_aspect_model.pt')
            #doconv architecture rest
            #torch.save(aspect_model,'./weights/latest_rest_doconv_aspect_model.pt')
            #xsep architecture rest
            #torch.save(aspect_model, './weights/latest_rest_xsepconv_aspect_model.pt')
            #2 xsep & 1 conv architecture rest
            #torch.save(aspect_model, './weights/latest_rest_xsepconv2_aspect_model.pt')
            #1 xsep & 2 conv architecture rest
            #torch.save(aspect_model,'./weights/latest_rest_xsepconv3_aspect_model.pt')
            # 1 xsep & 2 conv architecture lap
            #torch.save(aspect_model,'./weights/latest_lap_xsepconv3_aspect_model.pt')
            # 1 xsep & 2 conv architecture lap
            #torch.save(aspect_model,'./weights/latest_lap_xsepconv2_aspect_model.pt')
            # 3 xsep
            #torch.save(aspect_model,'./weights/latest_lap_xsepconv_aspect_model.pt')
            # doconv architecture lap
            #torch.save(aspect_model,'./weights/latest_lap_doconv_aspect_model.pt')
            # doconv architecture lap
            #torch.save(aspect_model,'./weights/latest_lap_doconv1_aspect_model.pt')
            #author architecture lap
            #torch.save(aspect_model, './weights/latest_lap_aspect_model.pt')
            # xsep rest doconv rq3
            #torch.save(aspect_model,'./weights/latest_lap_vd17da1_xsep_doconv_aspect_model.pt')
            # xsep rest doconv vd9
            #torch.save(aspect_model, './weights/latest_rest_vd9_xsep_doconv_aspect_model.pt')
            # xsep rest doconv vd9
            #torch.save(aspect_model,'./weights/latest_lap_vd9_xsep_doconv_aspect_model.pt')
            # xsep lap doconv vd17
            #torch.save(aspect_model,'./weights/latest_lap_vd17_xsep_doconv_aspect_model.pt')
            # xsep rest doconv vd17
            #torch.save(aspect_model,'./weights/latest_rest_vd17_xsep_doconv_aspect_model.pt')
            # xsep rest doconv vd29
            #torch.save(aspect_model,'./weights/latest_rest_vd29_xsep_doconv_aspect_model.pt')
            # xsep rest doconv vd29
            #torch.save(aspect_model,'./weights/latest_lap_vd29_xsep_doconv_aspect_model.pt')
            # xsep rest doconv vd17
            # torch.save(aspect_model,'./weights/latest_rest_vd17da_xsep_doconv_aspect_model.pt')
            # xsep lap doconv vd9 rq5
            #torch.save(aspect_model,'./weights/latest_lap_vd9_da_xsep_doconv_aspect_model.pt')
            # xsep rest doconv vd9 rq5
            #torch.save(aspect_model, './weights/latest_rest_vd9_da_xsep_doconv_aspect_model.pt')

            #architecture rest xsep
            #torch.save(aspect_model,'./weights/latest_rest_ae_aspect_model.pt')
            #doconv
            #torch.save(aspect_model, './weights/latest_rest_ae_doconv_aspect_model.pt')
            #xsep
            #torch.save(aspect_model, './weights/latest_rest_ae_xsep_aspect_model.pt')
            #xsep rest
            #torch.save(aspect_model,'./weights/latest_re_ae_xsep_aspect_model.pt')
            #doconv rest
            #torch.save(aspect_model, './weights/latest_re_ae_doconv_aspect_model.pt')
            #conv rest
            #torch.save(aspect_model, './weights/latest_re_ae_conv_aspect_model.pt')
            # xsep lap conv
            #torch.save(aspect_model,'./weights/latest_lap_ae_xsepconv_aspect_model.pt')
            # xsep lap doconv
            #torch.save(aspect_model,'./weights/latest_lap_ae_xsepdoconv_aspect_model.pt')
            # xsep lap xsep
            #torch.save(aspect_model, './weights/latest_lap_ae_xsepxsep_aspect_model.pt')
            # doconv xsep lap
            #torch.save(aspect_model,'./weights/latest_lap_ae_xsep_doconv_aspect_model.pt')
            # doconv rest doconv
            #torch.save(aspect_model,'./weights/latest_lap_ae_doconv_doconv_aspect_model.pt')
            # conv rest doconv
            #torch.save(aspect_model,'./weights/latest_lap_ae_conv_doconv_aspect_model.pt')
            # doconv xsep rest
            #torch.save(aspect_model,'./weights/latest_re_ae_xsep_doconv_aspect_model.pt')
            # xsep lap doconv
            #torch.save(aspect_model,'./weights/latest_lap_ae_xsep_doconv1_aspect_model.pt')
            # xsep rest doconv
            #torch.save(aspect_model,'./weights/latest_rest_ae_xsep_doconv1_aspect_model.pt')


            # authors architecture rest
            #torch.save(senti_model,'./weights/latest_rest_original_senti_model.pt')
            #doconv architecture rest
            #torch.save(senti_model,'./weights/latest_rest_doconv_senti_model.pt')
            #xsep architecture rest
            #torch.save(senti_model,'./weights/latest_rest_xsepconv_senti_model.pt')
            #2 xsep & 1 conv architecture rest
            #torch.save(senti_model, './weights/latest_rest_xsepconv2_senti_model.pt')
            # 1 xsep & 2 conv architecture rest
            #torch.save(senti_model, './weights/latest_rest_xsepconv3_senti_model.pt')
            # 1 xsep & 2 conv architecture lap
            #torch.save(senti_model,'./weights/latest_lap_xsepconv3_senti_model.pt')
            # 2 xsep & 1 conv architecture lap
            #torch.save(senti_model,'./weights/latest_lap_xsepconv2_senti_model.pt')
            # 3 xsep architecture
            #torch.save(senti_model,'./weights/latest_lap_xsepconv_senti_model.pt')
            # doconv architecture lap
            #torch.save(senti_model,'./weights/latest_lap_doconv_senti_model.pt')
            # doconv architecture lap correct
            #torch.save(senti_model, './weights/latest_lap_doconv1_senti_model.pt')
            #author architecture lap
            #torch.save(senti_model,'./weights/latest_doconv_senti_model.pt')
            # xsep lap doconv rq3
            #torch.save(senti_model,'./weights/latest_lap_vd17da1_xsep_doconv_senti_model.pt')
            # xsep rest doconv vd9
            #torch.save(senti_model,'./weights/latest_rest_vd9_xsep_doconv_senti_model.pt')
            # xsep rest doconv vd9
            #torch.save(senti_model,'./weights/latest_lap_vd9_xsep_doconv_senti_model.pt')
            # xsep lap doconv vd 17
            #torch.save(senti_model,'./weights/latest_lap_vd17_xsep_doconv_senti_model.pt')
            # xsep rest doconv vd17
            #torch.save(senti_model,'./weights/latest_rest_vd17_xsep_doconv_senti_model.pt')
            # xsep rest doconv vd29
            #torch.save(senti_model,'./weights/latest_rest_vd29_xsep_doconv_senti_model.pt')
            # xsep rest doconv vd29
            #torch.save(senti_model,'./weights/latest_lap_vd29_xsep_doconv_senti_model.pt')
            # xsep rest doconv vd17da
            # torch.save(senti_model,'./weights/latest_rest_vd17da_xsep_doconv_senti_model.pt')
            # xsep lap doconv rq5
            #torch.save(senti_model,'./weights/latest_lap_vd9_da_xsep_doconv_senti_model.pt')
            # xsep rest doconv rq5
            #torch.save(senti_model,'./weights/latest_rest_vd9_da_xsep_doconv_senti_model.pt')

            #architecture rest xsep author
            #torch.save(senti_model,'./weights/latest_rest_ae_senti_model.pt')
            #doconv
            #torch.save(senti_model,'./weights/latest_rest_ae_doconv_senti_model.pt')
            #xsep
            #torch.save(senti_model,'./weights/latest_rest_ae_xsep_senti_model.pt')
            #xsep rest
            #torch.save(senti_model,'./weights/latest_re_ae_xsep_senti_model.pt')
            # doconv rest
            #torch.save(senti_model, './weights/latest_re_ae_doconv_senti_model.pt')
            #conv rest
            #torch.save(senti_model, './weights/latest_re_ae_conv_senti_model.pt')
            # xsep lap conv
            #torch.save(senti_model,'./weights/latest_lap_ae_xsepconv_senti_model.pt')
            # xsep lap doconv
            #torch.save(senti_model,'./weights/latest_lap_ae_xsepdoconv_senti_model.pt')
            # xsep lap xsep
            #torch.save(senti_model,'./weights/latest_lap_ae_xsepxsep_senti_model.pt')
            # xsep rest doconv
            #torch.save(senti_model,'./weights/latest_lap_ae_xsep_doconv_senti_model.pt')
            # doconv rest doconv
            #torch.save(senti_model, './weights/latest_lap_ae_doconv_doconv_senti_model.pt')
            # conv rest doconv
            #torch.save(senti_model,'./weights/latest_lap_ae_conv_doconv_senti_model.pt')
            # xsep lap doconv
            #torch.save(senti_model,'./weights/latest_lap_ae_xsep_doconv1_senti_model.pt')
            # xsep rest doconv
            #torch.save(senti_model,'./weights/latest_rest_ae_xsep_doconv1_senti_model.pt')

            valid_loss, aspect_test_acc, _, pseudo_aspect_test_acc, _, aspect_pred_distribution = test(old_train_dataset_aspect, aspect_model, 'aspect')
            #authors architecture rest
            #with open('./Pickledata/latest_rest_original_train_dataset_aspect.pkl','wb') as f:
            #doconv architecture rest
            #with open('./Pickledata/latest_rest_doconv_train_dataset_aspect.pkl','wb') as f:
            #xsepconv architecture rest
            #with open('./Pickledata/latest_rest_xsepconv_train_dataset_aspect.pkl','wb') as f:
            # 2 xsepconv & 1 conv architecture rest
            #with open('./Pickledata/latest_rest_xsepconv2_train_dataset_aspect.pkl','wb') as f:
            # 1 xsepconv & 2 conv architecture rest
            #with open('./Pickledata/latest_rest_xsepconv3_train_dataset_aspect.pkl','wb') as f:
            # 1 xsepconv & 2 conv architecture lap
            #with open('./Pickledata/latest_lang_xsepconv3_train_dataset_aspect.pkl', 'wb') as f:
            # 2 xsepconv & 1 conv architecture lap
            #with open('./Pickledata/latest_lang_xsepconv2_train_dataset_aspect.pkl', 'wb') as f:
            # 3 xsepconv
            #with open('./Pickledata/latest_lap_xsepconv_train_dataset_aspect.pkl', 'wb') as f:
            # doconv architecture lap
            #with open('./Pickledata/latest_lap_doconv_train_dataset_aspect.pkl','wb') as f:
            # doconv architecture lap correct
            #with open('./Pickledata/latest_lap_doconv1_train_dataset_aspect.pkl','wb') as f:
            # author architecture lap
            #with open('./Pickledata/latest_lap_train_dataset_aspect.pkl','wb') as f:
            # xsep doconv lap rq3
            #with open('./Pickledata/latest_lap_vd17da1_xsep_doconv_dataset_aspect.pkl','wb') as f:
            # xsep doconv rest vd9
            #with open('./Pickledata/latest_rest_vd9_xsep_doconv_dataset_aspect.pkl', 'wb') as f:
            # xsep doconv lap vd9
            #with open('./Pickledata/latest_lap_vd9_xsep_doconv_dataset_aspect.pkl','wb') as f:
            # xsep doconv lap vd9
            #with open('./Pickledata/latest_lap_vd17_xsep_doconv_dataset_aspect.pkl','wb') as f:
            # xsep doconv lap vd9
            #with open('./Pickledata/latest_rest_vd17_xsep_doconv_dataset_aspect.pkl','wb') as f:
            # xsep doconv lap vd9
            #with open('./Pickledata/latest_rest_vd29_xsep_doconv_dataset_aspect.pkl','wb') as f:
            # xsep doconv lap vd9
            #with open('./Pickledata/latest_lap_vd29_xsep_doconv_dataset_aspect.pkl', 'wb') as f:
            # xsep doconv rest vd17
            # with open('./Pickledata/latest_rest_vd17da_xsep_doconv_dataset_aspect.pkl','wb') as f:
            # xsep doconv lap vd9
            #with open('./Pickledata/latest_lap_vd9_da_xsep_doconv_dataset_aspect.pkl','wb') as f:
            # xsep doconv rest vd9
            #with open('./Pickledata/latest_rest_vd9_da_xsep_doconv_dataset_aspect.pkl','wb') as f:

                #architecture xsep author ae
            #with open('./Pickledata/latest_rest_ae_train_dataset_aspect.pkl','wb') as f:
            # architecture xsep doconv ae
            #with open('./Pickledata/latest_rest_ae_doconv_dataset_aspect.pkl','wb') as f:
            #xsep
            #with open('./Pickledata/latest_rest_ae_xsep_dataset_aspect.pkl','wb') as f:
            #xsep rest
            #with open('./Pickledata/latest_re_ae_xsep_dataset_aspect.pkl','wb') as f:
            #doconv rest
            #with open('./Pickledata/latest_re_ae_doconv_dataset_aspect.pkl', 'wb') as f:
            #conv rest
            #with open('./Pickledata/latest_re_ae_conv_dataset_aspect.pkl','wb') as f:
            # xsep conv rest
            #with open('./Pickledata/latest_lap_ae_xsepconv_dataset_aspect.pkl','wb') as f:
            # xsep doconv rest
            #with open('./Pickledata/latest_lap_ae_xsepdoconv_dataset_aspect.pkl','wb') as f:
            # xsep doconv lap
            #with open('./Pickledata/latest_lap_ae_xsepxsep_dataset_aspect.pkl','wb') as f:
            # xsep doconv rest
            #with open('./Pickledata/latest_lap_ae_xsep_doconv_dataset_aspect.pkl','wb') as f:
            # doconv doconv rest
            #with open('./Pickledata/latest_lap_ae_doconv_doconv_dataset_aspect.pkl','wb') as f:
            # conv rest doconv
            #with open('./Pickledata/latest_lap_ae_conv_doconv_dataset_aspect.pkl','wb') as f:
            # xsep doconv lap
            #with open('./Pickledata/latest_lap_ae_xsep_doconv1_dataset_aspect.pkl','wb') as f:
            # xsep doconv rest
            #with open('./Pickledata/latest_rest_ae_xsep_doconv1_dataset_aspect.pkl','wb') as f:
                #pickle.dump(old_train_dataset_aspect, f)

            # print_info('Aspect Model', train_loss, aspect_test_acc, 0, pseudo_aspect_test_acc, 0)
            valid_loss, _, senti_test_acc, _, pseudo_senti_test_acc, senti_pred_distribution = test(old_train_dataset_senti, senti_model, 'senti')
            # print_info('Sentiment Model', train_loss, 0, senti_test_acc, 0, pseudo_senti_test_acc)

            # authors architecture rest
            #with open('./Pickledata/latest_rest_original_train_dataset_senti.pkl', 'wb') as f:
            #doconv architecture rest
            #with open('./Pickledata/latest_rest_doconv_train_dataset_senti.pkl', 'wb') as f:
            #xsepconv architecture rest
            #with open('./Pickledata/latest_rest_xsepconv_train_dataset_senti.pkl','wb') as f:
            # 2 xsepconv & 1 conv architecture rest
            #with open('./Pickledata/latest_rest_xsepconv2_train_dataset_senti.pkl','wb') as f:
            # 1 xsepconv & 2 conv architecture rest
            #with open('./Pickledata/latest_rest_xsepconv3_train_dataset_senti.pkl','wb') as f:
            # 1 xsepconv & 2 conv architecture lap
            #with open('./Pickledata/latest_lang_xsepconv3_train_dataset_senti.pkl', 'wb') as f:
            # 2 xsepconv & 1 conv architecture lap
            #with open('./Pickledata/latest_lang_xsepconv2_train_dataset_senti.pkl', 'wb') as f:
            # 3 xsepconv architecture lap
            #with open('./Pickledata/latest_lap_xsepconv_train_dataset_senti.pkl','wb') as f:
            #doconv
            #with open('./Pickledata/latest_lap_doconv_train_dataset_senti.pkl','wb') as f:
            #doconv correct
            #with open('./Pickledata/latest_lap_doconv1_train_dataset_senti.pkl','wb') as f:
            # author lap
            #with open('./Pickledata/latest_lap_train_dataset_senti.pkl','wb') as f:
            # xsep lap doconv rq3
            #with open('./Pickledata/latest_lapp_vd17da1_xsepdoconv_train_dataset_senti.pkl','wb') as f:
            # xsep rest doconv vd9
            # with open( './Pickledata/latest_rest_vd9_xsepdoconv_train_dataset_senti.pkl','wb') as f:
            # xsep lap doconv vd9
            #with open('./Pickledata/latest_lap_vd9_xsepdoconv_train_dataset_senti.pkl','wb') as f:
            # xsep lap doconv vd17
            #with open('./Pickledata/latest_lap_vd17_xsepdoconv_train_dataset_senti.pkl','wb') as f:
            # xsep lap doconv vd17
            #with open('./Pickledata/latest_rest_vd17_xsepdoconv_train_dataset_senti.pkl','wb') as f:
            # xsep lap doconv vd29
            #with open('./Pickledata/latest_rest_vd29_xsepdoconv_train_dataset_senti.pkl', 'wb') as f:
            # xsep lap doconv vd29
            #with open('./Pickledata/latest_lap_vd29_xsepdoconv_train_dataset_senti.pkl','wb') as f:
            # xsep lap doconv vd17 da
            # with open('./Pickledata/latest_lap_vd17da_xsepdoconv_train_dataset_senti.pkl','wb') as f:
            # xsep lap doconv vd9
            #with open('./Pickledata/latest_lap_vd9_da_xsepdoconv_train_dataset_senti.pkl','wb') as f:
            # xsep lap doconv vd9
            #with open('./Pickledata/latest_rest_vd9_da_xsepdoconv_train_dataset_senti.pkl', 'wb') as f:
            #architecture
            #with open('./Pickledata/latest_rest_ae_train_dataset_senti.pkl','wb') as f:
            #author doconv
            #with open('./Pickledata/latest_rest_ae_doconv_train_dataset_senti.pkl','wb') as f:
            #xsep
            #with open('./Pickledata/latest_rest_ae_xsep_train_dataset_senti.pkl','wb') as f:
            #doconv rest ae
            #with open('./Pickledata/latest_re_ae_xsep_train_dataset_senti.pkl', 'wb') as f:
            # doconv rest ae doconv
            #with open('./Pickledata/latest_re_ae_doconv_train_dataset_senti.pkl', 'wb') as f:
            # conv rest ae
            #with open('./Pickledata/latest_re_ae_conv_train_dataset_senti.pkl','wb') as f:
            # xsep lap conv
            #with open('./Pickledata/latest_lap_ae_xsepconv_train_dataset_senti.pkl','wb') as f:
            # xsep lap doconv
            #with open('./Pickledata/latest_lap_ae_xsepdoconv_train_dataset_senti.pkl','wb') as f:
            # xsep lap xsep
            #with open('./Pickledata/latest_lap_ae_xsepxsep_train_dataset_senti.pkl','wb') as f:
            # xsep rest doconv
            #with open('./Pickledata/latest_lap_ae_xsep_doconv_train_dataset_senti.pkl','wb') as f:
            # doconv rest doconv
            #with open('./Pickledata/latest_lap_ae_doconv_doconv_train_dataset_senti.pkl','wb') as f:
            # conv rest doconv
            #with open('./Pickledata/latest_lap_ae_conv_doconv_train_dataset_senti.pkl','wb') as f:
            # xsep lap doconv
            #with open('./Pickledata/latest_lap_ae_xsepdoconv1_train_dataset_senti.pkl','wb') as f:
            # xsep lap doconv
            #with open('./Pickledata/latest_rest_ae_xsepdoconv1_train_dataset_senti.pkl','wb') as f:
               #pickle.dump(old_train_dataset_senti, f)

            # The code is completely from the weakly supervised code https://github.com/teapot123/JASen and sklearn library
            target_names1 = ['Positive', 'Negative']
            classification_report1 = classification_report(np.array(1 - gt2), senti_pred_labels.numpy(),
                                                           target_names=target_names1,digits=4)
            print("classification_report", classification_report1)
            print("Confusion matrix Sentiment:",confusion_matrix(np.array(1 - gt2), senti_pred_labels.numpy()))

            print("Confusion matrix Aspect:",confusion_matrix(np.array(gt1), aspect_pred_labels.numpy()))
            target_names = ['Location', 'Drinks', 'Food', 'Ambience', 'Service']
            #target_names = ['Support', 'OS', 'Display', 'Battery', 'Company','Mouse','Software','Keyboard']
            classification_report = classification_report(np.array(gt1), aspect_pred_labels.numpy(),
                                                          target_names=target_names, digits=4)
            print("classification_report", classification_report)

            # The code is completely from the weakly supervised code https://github.com/teapot123/JASen
            aspect_target_scores = target_score(aspect_pred_distribution, 'aspect')
            aspect_pred_labels = aspect_pred_distribution.argmax(-1)
            senti_target_scores = target_score(senti_pred_distribution, 'senti')
            senti_pred_labels = senti_pred_distribution.argmax(-1)
            with open(os.path.join(dataset, 'prediction_lap.txt'),'w') as fout:
                fout.write("Aspect.pred\tAspect.gt\tSenti.pred\tSenti.gt\n")
                for k in range(0,len(aspect_pred_labels)):
                    fout.write(f"{aspect_pred_labels[k]}\t{old_gt1[k]}\t{1-senti_pred_labels[k]}\t{old_gt2[k]}\n")
            print(f"Results written to {dataset}/prediction.txt !")
            break


    

    


    
# The code is completely from the weakly supervised code https://github.com/teapot123/JASen
import torch
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from evaluate import get_emb
from evaluate import get_joint_temb
import pickle
from sklearn.metrics import *

kl_criterion = torch.nn.KLDivLoss()
dataset = './datasets/restaurant'
w_emb_file = './datasets/restaurant/emb_mix_w.txt'
t_emb_file = './datasets/restaurant/emb_mix_t.txt'

# dataset = './datasets/laptop'
# w_emb_file = './datasets/laptop/emb_mix_w.txt'
# t_emb_file = './datasets/laptop/emb_mix_t.txt'

joint_word_emb, vocabulary, vocabulary_inv = get_emb(os.path.join(dataset, w_emb_file))
joint_topic_emb, senti_topic, aspect_topic = get_joint_temb(vec_file=os.path.join(dataset, t_emb_file))


def target_score(logits, aspect):
    preds = torch.nn.Softmax(dim=-1)(logits)  # batch * class

    if aspect == 'aspect' and dataset == 'datasets/restaurant':
    #if aspect == 'aspect' and dataset == 'datasets/laptop':
        weight = preds ** 1.2  # / torch.sum(preds, dim=0)
    else:
        weight = preds ** 2 / torch.sum(preds, dim=0)

    return (weight.t() / torch.sum(weight, dim=1)).t()

def generate_batch(batch):
    label = torch.cat([entry[1].unsqueeze(0) for entry in batch])
    text = []
    for entry in batch:
        length = len(entry[0])
        tmp = F.pad(torch.tensor(entry[0]), (0, 100 - len(entry[0])), 'constant', 0).unsqueeze(0)
        text.append(tmp)
        for i in range(100):
            if tmp[0][i] >= len(vocabulary):
                print(tmp[i])

    gt1 = torch.from_numpy(np.array([entry[2] for entry in batch]))
    gt2 = torch.from_numpy(np.array([entry[3] for entry in batch]))

    text = torch.cat(text)

    return text, label, gt1, gt2

def test(data_, model, mode):
    loss = 0
    acc = 0
    pseudo_aspect_test_acc = 0
    pseudo_senti_test_acc = 0
    aspect_test_acc = 0
    senti_test_acc = 0
    data = DataLoader(data_, batch_size=128, collate_fn=generate_batch)
    # print("data_",data_)
    pred_distribution = []
    for text, cls, gt1, gt2 in data:
        # text, cls, gt1, gt2 = text.to(device), cls.to(device), gt1.to(device), gt2.to(device)
        with torch.no_grad():
            output = model(text)
            loss = kl_criterion(torch.log(F.softmax(output, dim=-1)), cls)

            if mode == 'joint':
                pseudo_aspect_test_acc += (
                            output.argmax(1) % len(aspect_topic) == cls.argmax(1) % len(aspect_topic)).sum().item()
                pseudo_senti_test_acc += (
                            output.argmax(1) / len(aspect_topic) == cls.argmax(1) / len(aspect_topic)).sum().item()
                aspect_test_acc += (output.argmax(1) % len(aspect_topic) == gt1).sum().item()
                senti_test_acc += (output.argmax(1) / len(aspect_topic) == (1 - gt2)).sum().item()

            elif mode == 'aspect':
                pseudo_aspect_test_acc += (output.argmax(1) == cls.argmax(1)).sum().item()
                aspect_test_acc += (output.argmax(1) == gt1).sum().item()
            elif mode == 'senti':
                pseudo_senti_test_acc += (output.argmax(1) == cls.argmax(1)).sum().item()
                senti_test_acc += (output.argmax(1) == (1 - gt2)).sum().item()

            pred_distribution.append(output)

    return loss / len(data_), aspect_test_acc / len(data_), senti_test_acc / len(data_), pseudo_aspect_test_acc / len(
        data_), pseudo_senti_test_acc / len(data_), torch.cat(pred_distribution, dim=0)

# authors architecture rest
# with open('./Pickledata/latest_rest_original_train_dataset_senti.pkl', 'wb') as f:
# doconv architecture rest
# with open('./Pickledata/latest_rest_doconv_train_dataset_senti.pkl', 'wb') as f:
# xsepconv architecture rest
# with open('./Pickledata/latest_rest_xsepconv_train_dataset_senti.pkl','wb') as f:
# 2 xsepconv & 1 conv architecture rest
# with open('./Pickledata/latest_rest_xsepconv2_train_dataset_senti.pkl','wb') as f:
# 1 xsepconv & 2 conv architecture rest
# with open('./Pickledata/latest_rest_xsepconv3_train_dataset_senti.pkl','wb') as f:
# 1 xsepconv & 2 conv architecture lap
# with open('./Pickledata/latest_lang_xsepconv3_train_dataset_senti.pkl', 'wb') as f:
# 2 xsepconv & 1 conv architecture lap
# with open('./Pickledata/latest_lang_xsepconv2_train_dataset_senti.pkl', 'wb') as f:
# 3 xsepconv architecture lap
# with open('./Pickledata/latest_lap_xsepconv_train_dataset_senti.pkl','wb') as f:
# doconv
# with open('./Pickledata/latest_lap_doconv_train_dataset_senti.pkl','wb') as f:
# doconv correct
# with open('./Pickledata/latest_lap_doconv1_train_dataset_senti.pkl','wb') as f:
# author lap
# with open('./Pickledata/latest_lap_train_dataset_senti.pkl','wb') as f:
# xsep lap doconv rq3
# with open('./Pickledata/latest_lapp_vd17da1_xsepdoconv_train_dataset_senti.pkl','wb') as f:
# xsep rest doconv vd9
# with open( './Pickledata/latest_rest_vd9_xsepdoconv_train_dataset_senti.pkl','wb') as f:
# xsep lap doconv vd9
# with open('./Pickledata/latest_lap_vd9_xsepdoconv_train_dataset_senti.pkl','wb') as f:
# xsep lap doconv vd17
# with open('./Pickledata/latest_lap_vd17_xsepdoconv_train_dataset_senti.pkl','wb') as f:
# xsep lap doconv vd17
# with open('./Pickledata/latest_rest_vd17_xsepdoconv_train_dataset_senti.pkl','wb') as f:
# xsep lap doconv vd29
# with open('./Pickledata/latest_rest_vd29_xsepdoconv_train_dataset_senti.pkl', 'wb') as f:
# xsep lap doconv vd29
# with open('./Pickledata/latest_lap_vd29_xsepdoconv_train_dataset_senti.pkl','wb') as f:
# xsep lap doconv vd17 da
# with open('./Pickledata/latest_lap_vd17da_xsepdoconv_train_dataset_senti.pkl','wb') as f:
# xsep lap doconv vd9
# with open('./Pickledata/latest_lap_vd9_da_xsepdoconv_train_dataset_senti.pkl','wb') as f:
#rest data augmentation vdcnn 9 doconv
with open('./Pickledata/latest_rest_vd9_da_xsep_doconv_dataset_aspect.pkl', 'rb') as f:
    old_train_dataset_aspect = pickle.load(f)

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
#xsep lap doconv rq3
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
with open('./Pickledata/latest_rest_vd9_da_xsepdoconv_train_dataset_senti.pkl', 'rb') as f:
    old_train_dataset_senti = pickle.load(f)

# authors architecture rest
# with open('./Pickledata/latest_rest_original_old_gt1.pkl','wb') as f:
# doconv architecture rest
# with open('./Pickledata/latest_rest_doconv_old_gt1.pkl','wb') as f:
# xsepconv architecture rest
# with open('./Pickledata/latest_rest_xsepconv_old_gt1.pkl','wb') as f:
# 2 xsepconv & 2 conv architecture rest
# with open('./Pickledata/latest_rest_xsepconv2_old_gt1.pkl','wb') as f:
# 1 xsepconv & 2 conv architecture rest
# with open('./Pickledata/latest_rest_xsepconv3_old_gt1.pkl','wb') as f:
# 1 xsepconv & 2 conv architecture lap
# with open('./Pickledata/latest_lap_xsepconv3_old_gt1.pkl','wb') as f:
# 2 xsepconv & 1 conv architecture lap
# with open('./Pickledata/latest_lap_xsepconv2_old_gt1.pkl','wb') as f:
# 3 xsep
# with open('./Pickledata/latest_lap_xsepconv_old_gt1.pkl','wb') as f:
# doconv architecture lap
# with open('./Pickledata/latest_lap_doconv_old_gt1.pkl','wb') as f:
# xsep doconv rest rq3
# with open('./Pickledata/latest_lap_vd17da1_xsep_doconv_old_gt1.pkl','wb') as f:
# xsep rest doconv vd9
# with open('./Pickledata/latest_rest_vd9_xsep_doconv_old_gt1.pkl','wb') as f:
# xsep lap doconv vd9
# with open('./Pickledata/latest_lap_vd9_xsep_doconv_old_gt1.pkl','wb') as f:
# xsep lap doconv vd17
# with open('./Pickledata/latest_lap_vd17_xsep_doconv_old_gt1.pkl','wb') as f:
# xsep rest doconv vd17
# with open('./Pickledata/latest_rest_vd17_xsep_doconv_old_gt1.pkl','wb') as f:
# xsep rest doconv vd29
# with open('./Pickledata/latest_rest_vd29_xsep_doconv_old_gt1.pkl','wb') as f:
# xsep rest doconv vd9
# with open('./Pickledata/latest_lap_vd29_xsep_doconv_old_gt1.pkl','wb') as f:
# xsep doconv rest vd17
# with open('./Pickledata/latest_rest_vd17da_xsep_doconv_old_gt1.pkl','wb') as f:
# xsep doconv rest rq5
# with open('./Pickledata/latest_lap_vd9_da_xsep_doconv_old_gt1.pkl','wb') as f:
# xsep doconv rest rq5
with open('./Pickledata/latest_rest_vd9_da_xsep_doconv_old_gt1.pkl', 'rb') as f:
    old_gt1 = pickle.load(f)

#with open('./Pickledata/latest_lap_old_gt2.pkl','wb') as f:
# xsep rest doconv rq3
#with open('./Pickledata/latest_lap_vd17da1_xsep_doconv_old_gt2.pkl','wb') as f:
# xsep rest doconv vd9
#with open('./Pickledata/latest_rest_vd9_xsep_doconv_old_gt2.pkl','wb') as f:
# xsep rest doconv vd9
# #with open('./Pickledata/latest_lap_vd9_xsep_doconv_old_gt2.pkl','wb') as f:
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
# xsep lap doconv rq5
#with open('./Pickledata/latest_lap_vd9_da_xsep_doconv_old_gt2.pkl','wb') as f:
# xsep rest doconv rq5
with open('./Pickledata/latest_rest_vd9_da_xsep_doconv_old_gt2.pkl', 'rb') as f:
    old_gt2 = pickle.load(f)


#authors architecture rest
#with open('./Pickledata/latest_rest_original_gt1.pkl','wb') as f:
#doconv architecture rest
#with open('./Pickledata/latest_rest_doconv_gt1.pkl', 'wb') as f:
#xsepconv architecture rest
#with open('./Pickledata/latest_rest_xsepconv_gt1.pkl', 'wb') as f:
#2 xsepconv & 1 conv2d architecture rest
#with open('./Pickledata/latest_rest_xsepconv2_gt1.pkl','wb') as f:
#1 xsepconv & 2 conv2d architecture lap
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
# doconv enh rest vdcnn 17
#with open('./Pickledata//latest_rest_vd29_xsepdoconv_gt1.pkl','wb') as f:
# doconv enh rest vdcnn 17
#with open('./Pickledata/latest_lap_vd29_xsepdoconv_gt1.pkl','wb') as f:
# xsep dconv vd17 rest
# with open('./Pickledata/latest_rest_vd17da_xsepdoconv_gt1.pkl','wb') as f:
###rq5
#with open('./Pickledata/latest_lap_vd9_da_xsepdoconv_gt1.pkl','wb') as f:
#rq5
with open('./Pickledata/latest_rest_vd9_da_xsepdoconv_gt1.pkl', 'rb') as f:
    gt1 = pickle.load(f)

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
with open('./Pickledata/latest_resttt_vd9_da_xsepdoconv_gt2.pkl', 'rb') as f:
    gt2 = pickle.load(f)

# authors architecture rest
# torch.load(aspect_model, './weights/latest_rest_original_aspect_model.pt')
# doconv architecture rest
# torch.load(aspect_model,'./weights/latest_rest_doconv_aspect_model.pt')
# xsep architecture rest
# torch.load(aspect_model, './weights/latest_rest_xsepconv_aspect_model.pt')
# 2 xsep & 1 conv architecture rest
# torch.load(aspect_model, './weights/latest_rest_xsepconv2_aspect_model.pt')
# 1 xsep & 2 conv architecture rest
# torch.load(aspect_model,'./weights/latest_rest_xsepconv3_aspect_model.pt')
# 1 xsep & 2 conv architecture lap
# torch.load(aspect_model,'./weights/latest_lap_xsepconv3_aspect_model.pt')
# 1 xsep & 2 conv architecture lap
# torch.load(aspect_model,'./weights/latest_lap_xsepconv2_aspect_model.pt')
# 3 xsep
# torch.load(aspect_model,'./weights/latest_lap_xsepconv_aspect_model.pt')
# doconv architecture lap
# torch.load(aspect_model,'./weights/latest_lap_doconv_aspect_model.pt')
# doconv architecture lap
# torch.load(aspect_model,'./weights/latest_lap_doconv1_aspect_model.pt')
# author architecture lap
# torch.load(aspect_model, './weights/latest_lap_aspect_model.pt')
# xsep rest doconv rq3
# torch.load(aspect_model,'./weights/latest_lap_vd17da1_xsep_doconv_aspect_model.pt')
# xsep rest doconv vd9
# torch.load(aspect_model, './weights/latest_rest_vd9_xsep_doconv_aspect_model.pt')
# xsep rest doconv vd9
# torch.load(aspect_model,'./weights/latest_lap_vd9_xsep_doconv_aspect_model.pt')
# xsep lap doconv vd17
# torch.load(aspect_model,'./weights/latest_lap_vd17_xsep_doconv_aspect_model.pt')
# xsep rest doconv vd17
# torch.load(aspect_model,'./weights/latest_rest_vd17_xsep_doconv_aspect_model.pt')
# xsep rest doconv vd29
# torch.load(aspect_model,'./weights/latest_rest_vd29_xsep_doconv_aspect_model.pt')
# xsep rest doconv vd29
# torch.load(aspect_model,'./weights/latest_lap_vd29_xsep_doconv_aspect_model.pt')
# xsep rest doconv vd17
# torch.load(aspect_model,'./weights/latest_rest_vd17da_xsep_doconv_aspect_model.pt')
# xsep lap doconv vd9 rq5
# torch.load(aspect_model,'./weights/latest_lap_vd9_da_xsep_doconv_aspect_model.pt')
# xsep rest doconv vd9 rq5
aspect_model = torch.load('./weights/latest_rest_vd9_da_xsep_doconv_aspect_model.pt')

# authors architecture rest
# torch.load(senti_model,'./weights/latest_rest_original_senti_model.pt')
# doconv architecture rest
# torch.load(senti_model,'./weights/latest_rest_doconv_senti_model.pt')
# xsep architecture rest
# torch.load(senti_model,'./weights/latest_rest_xsepconv_senti_model.pt')
# 2 xsep & 1 conv architecture rest
# torch.load(senti_model, './weights/latest_rest_xsepconv2_senti_model.pt')
# 1 xsep & 2 conv architecture rest
# torch.load(senti_model, './weights/latest_rest_xsepconv3_senti_model.pt')
# 1 xsep & 2 conv architecture lap
# torch.load(senti_model,'./weights/latest_lap_xsepconv3_senti_model.pt')
# 2 xsep & 1 conv architecture lap
# torch.load(senti_model,'./weights/latest_lap_xsepconv2_senti_model.pt')
# 3 xsep architecture
# torch.load(senti_model,'./weights/latest_lap_xsepconv_senti_model.pt')
# doconv architecture lap
# torch.load(senti_model,'./weights/latest_lap_doconv_senti_model.pt')
# doconv architecture lap correct
# torch.load(senti_model, './weights/latest_lap_doconv1_senti_model.pt')
# author architecture lap
# torch.load(senti_model,'./weights/latest_doconv_senti_model.pt')
# xsep lap doconv rq3
# torch.load(senti_model,'./weights/latest_lap_vd17da1_xsep_doconv_senti_model.pt')
# xsep rest doconv vd9
# torch.load(senti_model,'./weights/latest_rest_vd9_xsep_doconv_senti_model.pt')
# xsep rest doconv vd9
# torch.load(senti_model,'./weights/latest_lap_vd9_xsep_doconv_senti_model.pt')
# xsep lap doconv vd 17
# torch.load(senti_model,'./weights/latest_lap_vd17_xsep_doconv_senti_model.pt')
# xsep rest doconv vd17
# torch.load(senti_model,'./weights/latest_rest_vd17_xsep_doconv_senti_model.pt')
# xsep rest doconv vd29
# torch.load(senti_model,'./weights/latest_rest_vd29_xsep_doconv_senti_model.pt')
# xsep rest doconv vd29
# torch.load(senti_model,'./weights/latest_lap_vd29_xsep_doconv_senti_model.pt')
# xsep rest doconv vd17da
# torch.load(senti_model,'./weights/latest_rest_vd17da_xsep_doconv_senti_model.pt')
# xsep lap doconv rq5
# torch.load(senti_model,'./weights/latest_lap_vd9_da_xsep_doconv_senti_model.pt')
# xsep rest doconv rq5
senti_model = torch.load('./weights/latest_rest_vd9_da_xsep_doconv_senti_model.pt')
aspect_model.eval()
senti_model.eval()

# torch.loa(aspect_model,'/home/sowmya/Documents/Thesis/CorrectCode/WeakABSA/weights/aspect_model.pt')
# print(senti_model)
# torch.load(senti_model,'/home/sowmya/Documents/Thesis/CorrectCode/WeakABSA/weights/senti_model.pt')
valid_loss, aspect_test_acc, _, pseudo_aspect_test_acc, _, aspect_pred_distribution = test(old_train_dataset_aspect,
                                                                                               aspect_model, 'aspect')
# print_info('Aspect Model', train_loss, aspect_test_acc, 0, pseudo_aspect_test_acc, 0)
valid_loss, _, senti_test_acc, _, pseudo_senti_test_acc, senti_pred_distribution = test(old_train_dataset_senti,
                                                                                            senti_model, 'senti')
# print_info('Sentiment Model', train_loss, 0, senti_test_acc, 0, pseudo_senti_test_acc)

aspect_target_scores = target_score(aspect_pred_distribution, 'aspect')
aspect_pred_labels = aspect_pred_distribution.argmax(-1)
senti_target_scores = target_score(senti_pred_distribution, 'senti')
senti_pred_labels = senti_pred_distribution.argmax(-1)
with open(os.path.join(dataset, 'prediction1.txt'), 'w') as fout:
    fout.write("Aspect.pred\tAspect.gt\tSenti.pred\tSenti.gt\n")
    for k in range(len(aspect_pred_labels)):
        fout.write(f"{aspect_pred_labels[k]}\t{old_gt1[k]}\t{1 - senti_pred_labels[k]}\t{old_gt2[k]}\n")
print(f"Results written to {dataset}/prediction_oldmodel_NS.txt !")

print("Confusion matrix Aspect:",confusion_matrix(1- np.array(gt2), senti_pred_labels.numpy()))
target_names1 = ['Positive', 'Negative']
classification_report1 = classification_report(1 - np.array(gt2), senti_pred_labels.numpy(),
                                                               target_names=target_names1,digits=4)
print("classification_report", classification_report1)


print("Confusion matrix Aspect:",confusion_matrix(np.array(gt1), aspect_pred_labels.numpy()))
target_names = ['Location', 'Drinks', 'Food', 'Ambience', 'Service']
#target_names = ['Support', 'OS', 'Display', 'Battery', 'Company','Mouse','Software','Keyboard']
classification_report = classification_report(np.array(gt1), aspect_pred_labels.numpy(),
                                                              target_names=target_names, digits=4)
print("classification_report", classification_report)
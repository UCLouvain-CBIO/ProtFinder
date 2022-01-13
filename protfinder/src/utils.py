import torch
from ax import optimize
from loss import multilabel_loss, get_correct

from tqdm import tnrange, tqdm
from math import log

def hyperparam_opt(train_dl, test_dl, model, optim, device, batch_size):
    
    print('Running Hyperparameter Optimization')
    
    def objective(params):

        with torch.autograd.set_detect_anomaly(True):

            thresh = list()
            for i in range(28):
                thresh.append(params[f't{i+1}'])

            for epoch in range(params['epochs']):
                
                print(f'Epoch: {epoch}')
                epoch_loss = 0.0
                correct = 0
                total = 0
                flag = False
                model.train()

                for X, y in tqdm(train_dl):
                    if y.shape[0] == batch_size:
                        
                        pred = model(X, batch_size)
                        
                        loss = multilabel_loss(pred, y)

                        n_correct,_,_ = get_correct(pred, y, device=device, thresh=thresh)
                        correct += n_correct
                        total += batch_size

                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        epoch_loss += loss.item()

            
            model.eval()
            epoch_loss = 0.0
            correct = 0
            total = 0
            sens_total = 0
            spec_total = 0
            auc_total = 0
            mcc_total = 0

            target_list = list()
            pred_list = list()

            for X, y in tqdm(test_dl):
                if y.shape[0] == batch_size:
                    pred = model(X, batch_size)
                    loss = multilabel_loss(pred, y)

                    n_correct, _, (sens, spec, auc, mcc) = get_correct(pred, y, device=device, flag=flag, thresh=thresh)
            
                    correct += n_correct
                    total += batch_size
                    try:
                        spec_total += spec
                        sens_total += sens
                        auc_total += auc
                        mcc_total += mcc
                    except:
                        pass
                    target_list.append(y)
                    pred_list.append(pred)
                    epoch_loss += loss.item()
            


            if spec is not None and sens is not None and mcc is not None and auc is not None:
                sens = sens_total/total
                spec = spec_total/total
                auc = auc_total/total
                mcc = mcc_total/total
                
        return auc        
    
    best_params, best_vals , _, _ = optimize(
        parameters=[
            {
                'name': 'epochs',
                'type':'range',
                'bounds': [1, 30]
            },
            {
                'name': 't1',
                'type':'range',
                'bounds': [0.5, 0.9]
            },
            {
                'name': 't2',
                'type':'range',
                'bounds': [0.5, 0.8]
            },
            {
                'name': 't3',
                'type':'range',
                'bounds': [0.5, 0.9]
            },
            {
                'name': 't4',
                'type':'range',
                'bounds': [0.5, 0.9]
            },
            {
                'name': 't5',
                'type':'range',
                'bounds': [0.45, 0.55]
            },
            {
                'name': 't6',
                'type':'range',
                'bounds': [0.45, 0.55]
            },
            {
                'name': 't7',
                'type':'range',
                'bounds': [0.25, 0.75]
            },
            {
                'name': 't8',
                'type':'range',
                'bounds': [0.25, 0.75]
            },
            {
                'name': 't9',
                'type':'range',
                'bounds': [0.45, 0.55]
            },
            {
                'name': 't10',
                'type':'range',
                'bounds': [0.35, 0.6]
            },
            {
                'name': 't11',
                'type':'range',
                'bounds': [0.25, 0.75]
            },
            {
                'name': 't12',
                'type':'range',
                'bounds': [0.2, 0.5]
            },
            {
                'name': 't13',
                'type':'range',
                'bounds': [0.45, 0.55]
            },
            {
                'name': 't14',
                'type':'range',
                'bounds': [0.45, 0.55]
            },
            {
                'name': 't15',
                'type':'range',
                'bounds': [0.45, 0.55]
            },
            {
                'name': 't16',
                'type':'range',
                'bounds': [0.45, 0.55]
            },
            {
                'name': 't17',
                'type':'range',
                'bounds': [0.45, 0.55]
            },
            {
                'name': 't18',
                'type':'range',
                'bounds': [0.35, 0.75]
            },
            {
                'name': 't19',
                'type':'range',
                'bounds': [0.45, 0.55]
            },
            {
                'name': 't20',
                'type':'range',
                'bounds': [0.5, 0.65]
            },
            {
                'name': 't21',
                'type':'range',
                'bounds': [0.4, 0.65]
            },
            {
                'name': 't22',
                'type':'range',
                'bounds': [0.3, 0.75]
            },
            {
                'name': 't23',
                'type':'range',
                'bounds': [0.25, 0.75]
            },
            {
                'name': 't24',
                'type':'range',
                'bounds': [0.4, 0.6]
            },
            {
                'name': 't25',
                'type':'range',
                'bounds': [0.25, 0.75]
            },
            {
                'name': 't26',
                'type':'range',
                'bounds': [0.4, 0.6]
            },
            {
                'name': 't27',
                'type':'range',
                'bounds': [0.25, 0.75]
            },
            {
                'name': 't28',
                'type':'range',
                'bounds': [0.25, 0.75]
            }
        ],
        evaluation_function=objective,
        total_trials=5
    )
    print(f'Best Parameters - \n {best_params}')
    print(f'Best Values - \n {best_vals}')

def merge_class(main_l, new_l):
    if len(main_l) == 0:
        return new_l
    else:
        for i in range(len(main_l)):
            for j in range(4):
                main_l[i][j] += new_l[i][j]
        return main_l

def merge1(d1, d2):
    if len(d1) == 0:
        return d2
    for key, val in d2.items():
        try:
            d1[key] += val
        except:
            d1[key] = val
    return d1

def merge2(d1, d2):
    if len(d1) == 0:
        return d2
    for k, vals in d2.items():
        try:
            d1[k]
        except:
            d1[k] = dict()
        for key, val in vals.items():
            try:
                d1[k][key] += val
            except:
                d1[k][key] = val
    return d1

def get_distwise_mcc(meta_info):
    meta = dict()
    for key, val in meta_info.items():
        TP = val['tp']
        FP = val['fp']
        TN = val['tn']
        FN = val['fn']
        MCC = ((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)

        meta[key] = MCC
        
    return meta

def get_classwise_scores(scores):
    output = list()
    for vals in scores:
        tp = vals[0]
        fp = vals[1]
        tn = vals[2]
        fn = vals[3]
        try:
            mcc = ((tp*tn)-(fp*fn))/(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)
        except:
            mcc = -1
        output.append(mcc)
    return output
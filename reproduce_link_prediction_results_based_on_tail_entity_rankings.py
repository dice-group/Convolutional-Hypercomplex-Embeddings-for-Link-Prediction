from util.helper_classes import Reproduce
from models.ensemble import Ensemble

run_WN18RR = True
run_FB15K_237 = True
run_UMLS = True
run_Kinship = True

#################### Description of PretrainedModels.zip ##########################################
# 1. PretrainedModels folder contains (WN18RR, FB15K-237, Kinship and UMLS) named folders.
# 2. Each folder contains (1. OnlyTrainSplit, QMult, OMult, ConvQ and ConvO) named folders.
# 3. (1. OnlyTrainSplit) named folders (QMult_train, OMult_train, ConvQ_train and ConvO_train) named folders.
# 4. Folders with _train provide models that are trained on the train splits of datasets others trained on train+valid splits.

#################### Description of pretrained models ##########################################
# 1. We provide info.log, loss_per_epoch.csv, model.pt and settings.json files
if run_WN18RR:
    kg_path = 'KGs/WN18RR'
    print('###########################################{0}##################################################'.format(
        kg_path))

    Reproduce().reproduce(model_path='PretrainedModels/WN18RR/QMult', data_path="%s/" % kg_path, model_name='QMult',
                          tail_pred_constraint=True)
    Reproduce().reproduce(model_path='PretrainedModels/WN18RR/ConvQ', data_path="%s/" % kg_path, model_name='ConvQ',
                          tail_pred_constraint=True)
    Reproduce().reproduce(model_path='PretrainedModels/WN18RR/OMult', data_path="%s/" % kg_path, model_name='OMult',
                          tail_pred_constraint=True)
    Reproduce().reproduce(model_path='PretrainedModels/WN18RR/ConvO', data_path="%s/" % kg_path, model_name='ConvO',
                          tail_pred_constraint=True)

    # ConvQ ConvO OMult Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvQ', model_name='ConvQ'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvO', model_name='ConvO'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/OMult', model_name='OMult')),
        data_path="%s/" % kg_path, tail_pred_constraint=True)

    # ConvQ ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvQ', model_name='ConvQ'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, tail_pred_constraint=True)

    # QMult OMult Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/OMult', model_name='OMult')),
        data_path="%s/" % kg_path, tail_pred_constraint=True)
    # QMult ConvQ Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvQ', model_name='ConvQ')),
        data_path="%s/" % kg_path, tail_pred_constraint=True)
    # QMult ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, tail_pred_constraint=True)
    # OMult ConvQ Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/OMult', model_name='OMult'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvQ', model_name='ConvQ')),
        data_path="%s/" % kg_path, tail_pred_constraint=True)
    # OMult ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/OMult', model_name='OMult'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, tail_pred_constraint=True)

    print('###########################################{0}##################################################'.format(
        kg_path))
if run_FB15K_237:
    kg_path = 'KGs/FB15k-237'
    print('###########################################{0}##################################################'.format(
        kg_path))
    Reproduce().reproduce(model_path='PretrainedModels/FB15K-237/QMult', data_path="%s/" % kg_path, model_name='QMult',
                          tail_pred_constraint=True)
    Reproduce().reproduce(model_path='PretrainedModels/FB15K-237/ConvQ', data_path="%s/" % kg_path, model_name='ConvQ',
                          tail_pred_constraint=True)
    Reproduce().reproduce(model_path='PretrainedModels/FB15K-237/OMult', data_path="%s/" % kg_path, model_name='OMult',
                          tail_pred_constraint=True)
    Reproduce().reproduce(model_path='PretrainedModels/FB15K-237/ConvO', data_path="%s/" % kg_path, model_name='ConvO',
                          tail_pred_constraint=True)

    # ConvQ ConvO OMult Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvQ', model_name='ConvQ'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvO', model_name='ConvO'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/OMult', model_name='OMult')),
        data_path="%s/" % kg_path, tail_pred_constraint=True)

    # ConvQ ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvQ', model_name='ConvQ'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, tail_pred_constraint=True)

    # QMult OMult Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/OMult', model_name='OMult')),
        data_path="%s/" % kg_path, tail_pred_constraint=True)
    # QMult ConvQ Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvQ', model_name='ConvQ')),
        data_path="%s/" % kg_path, tail_pred_constraint=True)
    # QMult ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, tail_pred_constraint=True)
    # OMult ConvQ Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/OMult', model_name='OMult'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvQ', model_name='ConvQ')),
        data_path="%s/" % kg_path, tail_pred_constraint=True)
    # OMult ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/OMult', model_name='OMult'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, tail_pred_constraint=True)

    print('###########################################{0}##################################################'.format(
        kg_path))

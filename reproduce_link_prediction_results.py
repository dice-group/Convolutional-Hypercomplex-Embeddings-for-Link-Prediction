from util.helper_classes import Reproduce
from models.ensemble import Ensemble

run_FB15K = True
run_WN18 = True
run_WN18RR = True
run_FB15K_237 = True
run_YAGO_3_10 = True
run_UMLS = True
run_Kinship = True

#################### Description of PretrainedModels.zip ##########################################
# 1. PretrainedModels folder contains (WN18RR, FB15K-237, Kinship and UMLS) named folders.
# 2. Each folder contains (1. OnlyTrainSplit, QMult, OMult, ConvQ and ConvO) named folders.
# 3. (1. OnlyTrainSplit) named folders (QMult_train, OMult_train, ConvQ_train and ConvO_train) named folders.
# 4. Folders with _train provide models that are trained on the train splits of datasets others trained on train+valid splits.

#################### Description of pretrained models ##########################################
# 1. We provide info.log, loss_per_epoch.csv, model.pt and settings.json files

if run_YAGO_3_10:
    kg_path = 'KGs/YAGO3-10'
    print('###########################################{0}##################################################'.format(
        kg_path))
    Reproduce().reproduce(model_path='PretrainedModels/YAGO3-10/QMult', data_path="%s/" % kg_path, model_name='QMult',
                          per_rel_flag_=True)
    Reproduce().reproduce(model_path='PretrainedModels/YAGO3-10/OMult', data_path="%s/" % kg_path, model_name='OMult',
                          per_rel_flag_=True)
    Reproduce().reproduce(model_path='PretrainedModels/YAGO3-10/ConvQ', data_path="%s/" % kg_path, model_name='ConvQ',
                          per_rel_flag_=True)

if run_FB15K:
    kg_path = 'KGs/FB15k'
    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))

    # Takes about 3 minutes.
    Reproduce().reproduce(model_path='PretrainedModels/FB15K/QMult', data_path="%s/" % kg_path, model_name='QMult')
    # Takes about 3 minutes.
    Reproduce().reproduce(model_path='PretrainedModels/FB15K/ConvQ', data_path="%s/" % kg_path, model_name='ConvQ')
    # Takes about X minutes.
    Reproduce().reproduce(model_path='PretrainedModels/FB15K/OMult', data_path="%s/" % kg_path, model_name='OMult')
    # Takes about X minutes.
    Reproduce().reproduce(model_path='PretrainedModels/FB15K/ConvO', data_path="%s/" % kg_path, model_name='ConvO')

if run_WN18:
    kg_path = 'KGs/WN18'
    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))

    # Takes about X minutes.
    Reproduce().reproduce(model_path='PretrainedModels/WN18/QMult', data_path="%s/" % kg_path, model_name='QMult')
    # Takes about X minutes.
    Reproduce().reproduce(model_path='PretrainedModels/WN18/ConvQ', data_path="%s/" % kg_path, model_name='ConvQ')
    # Takes about X minutes.
    Reproduce().reproduce(model_path='PretrainedModels/WN18/OMult', data_path="%s/" % kg_path, model_name='OMult')
    # Takes about X minutes.
    Reproduce().reproduce(model_path='PretrainedModels/WN18/ConvO', data_path="%s/" % kg_path, model_name='ConvO')

if run_WN18RR:
    kg_path = 'KGs/WN18RR'
    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))
    # Takes about X minutes.
    Reproduce().reproduce(model_path='PretrainedModels/WN18RR/QMult', data_path="%s/" % kg_path, model_name='QMult',
                          per_rel_flag_=True)
    # Takes about X minutes.
    Reproduce().reproduce(model_path='PretrainedModels/WN18RR/ConvQ', data_path="%s/" % kg_path, model_name='ConvQ',
                          per_rel_flag_=True)
    # Takes about X minutes.
    Reproduce().reproduce(model_path='PretrainedModels/WN18RR/OMult', data_path="%s/" % kg_path, model_name='OMult',
                          per_rel_flag_=True)
    # Takes about X minutes.
    Reproduce().reproduce(model_path='PretrainedModels/WN18RR/ConvO', data_path="%s/" % kg_path, model_name='ConvO',
                          per_rel_flag_=True)

    # Takes about X minutes.
    # ConvQ ConvO OMult Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvQ', model_name='ConvQ'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvO', model_name='ConvO'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/OMult', model_name='OMult')),
        data_path="%s/" % kg_path, per_rel_flag_=True)

    # Takes about X minutes.
    # ConvQ ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvQ', model_name='ConvQ'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, per_rel_flag_=False)

    # Takes about X minutes.
    # QMult OMult Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/OMult', model_name='OMult')),
        data_path="%s/" % kg_path, per_rel_flag_=False)

    # Takes about X minutes.
    # QMult ConvQ Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvQ', model_name='ConvQ')),
        data_path="%s/" % kg_path, per_rel_flag_=False)

    # Takes about X minutes.
    # QMult ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, per_rel_flag_=False)

    # Takes about X minutes.
    # OMult ConvQ Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/OMult', model_name='OMult'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvQ', model_name='ConvQ')),
        data_path="%s/" % kg_path, per_rel_flag_=False)

    # Takes about X minutes.
    # OMult ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/OMult', model_name='OMult'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, per_rel_flag_=False)

    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))

if run_FB15K_237:
    kg_path = 'KGs/FB15k-237'
    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))

    # Takes about X minutes.
    Reproduce().reproduce(model_path='PretrainedModels/FB15K-237/QMult', data_path="%s/" % kg_path, model_name='QMult',
                          per_rel_flag_=False)
    # Takes about X minutes.
    Reproduce().reproduce(model_path='PretrainedModels/FB15K-237/ConvQ', data_path="%s/" % kg_path, model_name='ConvQ',
                          per_rel_flag_=False)
    # Takes about X minutes.
    Reproduce().reproduce(model_path='PretrainedModels/FB15K-237/OMult', data_path="%s/" % kg_path, model_name='OMult',
                          per_rel_flag_=False)
    # Takes about X minutes.
    Reproduce().reproduce(model_path='PretrainedModels/FB15K-237/ConvO', data_path="%s/" % kg_path, model_name='ConvO',
                          per_rel_flag_=False)

    # Takes about X minutes.
    # ConvQ ConvO OMult Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvQ', model_name='ConvQ'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvO', model_name='ConvO'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/OMult', model_name='OMult')),
        data_path="%s/" % kg_path, per_rel_flag_=False)

    # Takes about X minutes.
    # ConvQ ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvQ', model_name='ConvQ'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, per_rel_flag_=False)

    # Takes about X minutes.
    # QMult OMult Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/OMult', model_name='OMult')),
        data_path="%s/" % kg_path, per_rel_flag_=False)

    # Takes about X minutes.
    # QMult ConvQ Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvQ', model_name='ConvQ')),
        data_path="%s/" % kg_path, per_rel_flag_=False)

    # Takes about X minutes.
    # QMult ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, per_rel_flag_=False)

    # Takes about X minutes.
    # OMult ConvQ Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/OMult', model_name='OMult'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvQ', model_name='ConvQ')),
        data_path="%s/" % kg_path, per_rel_flag_=False)
    # Takes about X minutes.
    # OMult ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/OMult', model_name='OMult'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, per_rel_flag_=False)

    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))


if run_UMLS:
    kg_path = 'KGs/UMLS'
    print('###########################################{0}##################################################'.format(
        kg_path))
    Reproduce().reproduce(model_path='PretrainedModels/UMLS/QMult', data_path="%s/" % kg_path, model_name='QMult',
                          per_rel_flag_=False)
    Reproduce().reproduce(model_path='PretrainedModels/UMLS/ConvQ', data_path="%s/" % kg_path, model_name='ConvQ',
                          per_rel_flag_=False)
    Reproduce().reproduce(model_path='PretrainedModels/UMLS/OMult', data_path="%s/" % kg_path, model_name='OMult',
                          per_rel_flag_=False)
    Reproduce().reproduce(model_path='PretrainedModels/UMLS/ConvO', data_path="%s/" % kg_path, model_name='ConvO',
                          per_rel_flag_=False)
    # ConvQ ConvO OMult Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/UMLS/ConvQ', model_name='ConvQ'),
                       Reproduce().load_model(model_path='PretrainedModels/UMLS/ConvO', model_name='ConvO'),
                       Reproduce().load_model(model_path='PretrainedModels/UMLS/OMult', model_name='OMult')),
        data_path="%s/" % kg_path, per_rel_flag_=False)

    # ConvQ ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/UMLS/ConvQ', model_name='ConvQ'),
                       Reproduce().load_model(model_path='PretrainedModels/UMLS/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, per_rel_flag_=False)

    # QMult OMult Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/UMLS/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/UMLS/OMult', model_name='OMult')),
        data_path="%s/" % kg_path, per_rel_flag_=False)
    # QMult ConvQ Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/UMLS/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/UMLS/ConvQ', model_name='ConvQ')),
        data_path="%s/" % kg_path, per_rel_flag_=False)
    # QMult ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/UMLS/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/UMLS/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, per_rel_flag_=False)
    # OMult ConvQ Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/UMLS/OMult', model_name='OMult'),
                       Reproduce().load_model(model_path='PretrainedModels/UMLS/ConvQ', model_name='ConvQ')),
        data_path="%s/" % kg_path, per_rel_flag_=False)
    # OMult ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/UMLS/OMult', model_name='OMult'),
                       Reproduce().load_model(model_path='PretrainedModels/UMLS/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, per_rel_flag_=False)
    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))

if run_Kinship:
    kg_path = 'KGs/KINSHIP'
    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))

    Reproduce().reproduce(model_path='PretrainedModels/Kinship/QMult', data_path="%s/" % kg_path, model_name='QMult',
                          per_rel_flag_=False)
    Reproduce().reproduce(model_path='PretrainedModels/Kinship/ConvQ', data_path="%s/" % kg_path, model_name='ConvQ',
                          per_rel_flag_=False)
    Reproduce().reproduce(model_path='PretrainedModels/Kinship/OMult', data_path="%s/" % kg_path, model_name='OMult',
                          per_rel_flag_=False)
    Reproduce().reproduce(model_path='PretrainedModels/Kinship/ConvO', data_path="%s/" % kg_path, model_name='ConvO',
                          per_rel_flag_=False)
    # ConvQ ConvO OMult Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/Kinship/ConvQ', model_name='ConvQ'),
                       Reproduce().load_model(model_path='PretrainedModels/Kinship/ConvO', model_name='ConvO'),
                       Reproduce().load_model(model_path='PretrainedModels/Kinship/OMult', model_name='OMult')),
        data_path="%s/" % kg_path, per_rel_flag_=False)

    # ConvQ ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/Kinship/ConvQ', model_name='ConvQ'),
                       Reproduce().load_model(model_path='PretrainedModels/Kinship/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, per_rel_flag_=False)

    # QMult OMult Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/Kinship/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/Kinship/OMult', model_name='OMult')),
        data_path="%s/" % kg_path, per_rel_flag_=False)
    # QMult ConvQ Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/Kinship/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/Kinship/ConvQ', model_name='ConvQ')),
        data_path="%s/" % kg_path, per_rel_flag_=False)
    # QMult ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/Kinship/QMult', model_name='QMult'),
                       Reproduce().load_model(model_path='PretrainedModels/Kinship/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, per_rel_flag_=False)
    # OMult ConvQ Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/Kinship/OMult', model_name='OMult'),
                       Reproduce().load_model(model_path='PretrainedModels/Kinship/ConvQ', model_name='ConvQ')),
        data_path="%s/" % kg_path, per_rel_flag_=False)
    # OMult ConvO Ensemble
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/Kinship/OMult', model_name='OMult'),
                       Reproduce().load_model(model_path='PretrainedModels/Kinship/ConvO', model_name='ConvO')),
        data_path="%s/" % kg_path, per_rel_flag_=False)
    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))

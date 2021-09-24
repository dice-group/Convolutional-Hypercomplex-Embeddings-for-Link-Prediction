from util.helper_classes import Reproduce
from models.ensemble import Ensemble

run_WN18RR = True
run_FB15K_237 = True
run_YAGO_3_10 = True
run_Kinship = True
run_UMLS = True
run_FB15K = True
run_WN18 = True
# To predict only tail entity, set it to true
tail_pred_constraint = False
# To apply range constraint, set it True. This is ongoing work.
apply_range_constraint = True
# To show link prediction results via ensemble models, set it True.
show_ensembles = False
if run_FB15K_237:
    kg_path = 'KGs/FB15k-237'
    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))
    Reproduce().reproduce(model_path='PretrainedModels/FB15K-237/QMultBatch', data_path="%s/" % kg_path,
                          model_name='QMultBatch', tail_pred_constraint=tail_pred_constraint,
                          apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/FB15K-237/OMultBatch', data_path="%s/" % kg_path,
                          model_name='OMultBatch', tail_pred_constraint=tail_pred_constraint,
                          apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/FB15K-237/ConvQBatch', data_path="%s/" % kg_path,
                          model_name='ConvQBatch', tail_pred_constraint=tail_pred_constraint,
                          apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/FB15K-237/ConvOBatch', data_path="%s/" % kg_path,
                          model_name='ConvOBatch', tail_pred_constraint=tail_pred_constraint,
                          apply_range_constraint=apply_range_constraint)
    if show_ensembles:
        Reproduce().reproduce_ensemble(
            model=Ensemble(
                Reproduce().load_model(model_path='PretrainedModels/FB15K-237/QMultBatch', model_name='QMultBatch'),
                Reproduce().load_model(model_path='PretrainedModels/FB15K-237/OMultBatch', model_name='OMultBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(
                Reproduce().load_model(model_path='PretrainedModels/FB15K-237/QMultBatch', model_name='QMultBatch'),
                Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvQBatch', model_name='ConvQBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(
                Reproduce().load_model(model_path='PretrainedModels/FB15K-237/QMultBatch', model_name='QMultBatch'),
                Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvOBatch', model_name='ConvOBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(
                Reproduce().load_model(model_path='PretrainedModels/FB15K-237/OMultBatch', model_name='OMultBatch'),
                Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvQBatch', model_name='ConvQBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(
                Reproduce().load_model(model_path='PretrainedModels/FB15K-237/OMultBatch', model_name='OMultBatch'),
                Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvOBatch', model_name='ConvOBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(
                Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvQBatch', model_name='ConvQBatch'),
                Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConvOBatch', model_name='ConvOBatch'),
                Reproduce().load_model(model_path='PretrainedModels/FB15K-237/OMultBatch', model_name='OMultBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))
if run_YAGO_3_10:
    kg_path = 'KGs/YAGO3-10'
    print('###########################################{0}##################################################'.format(
        kg_path))
    Reproduce().reproduce(model_path='PretrainedModels/YAGO3-10/QMult', data_path="%s/" % kg_path, model_name='QMult',
                          tail_pred_constraint=tail_pred_constraint, apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/YAGO3-10/OMult', data_path="%s/" % kg_path, model_name='OMult',
                          tail_pred_constraint=tail_pred_constraint, apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/YAGO3-10/ConvQ', data_path="%s/" % kg_path, model_name='ConvQ',
                          tail_pred_constraint=tail_pred_constraint, apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/YAGO3-10/ConvO', data_path="%s/" % kg_path, model_name='ConvO',
                          tail_pred_constraint=tail_pred_constraint, apply_range_constraint=apply_range_constraint)
    if show_ensembles:
        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/YAGO3-10/QMult', model_name='QMult'),
                           Reproduce().load_model(model_path='PretrainedModels/YAGO3-10/OMult', model_name='OMult')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/YAGO3-10/QMult', model_name='QMult'),
                           Reproduce().load_model(model_path='PretrainedModels/YAGO3-10/ConvQ', model_name='ConvQ')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/YAGO3-10/QMult', model_name='QMult'),
                           Reproduce().load_model(model_path='PretrainedModels/YAGO3-10/ConvO', model_name='ConvO')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/YAGO3-10/OMult', model_name='OMult'),
                           Reproduce().load_model(model_path='PretrainedModels/YAGO3-10/ConvQ', model_name='ConvQ')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/YAGO3-10/OMult', model_name='OMult'),
                           Reproduce().load_model(model_path='PretrainedModels/YAGO3-10/ConvO', model_name='ConvO')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/YAGO3-10/ConvQ', model_name='ConvQ'),
                           Reproduce().load_model(model_path='PretrainedModels/YAGO3-10/ConvO', model_name='ConvO'),
                           Reproduce().load_model(model_path='PretrainedModels/YAGO3-10/OMult', model_name='OMult')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)
if run_WN18RR:
    kg_path = 'KGs/WN18RR'
    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))
    Reproduce().reproduce(model_path='PretrainedModels/WN18RR/QMult', data_path="%s/" % kg_path, model_name='QMult',
                          tail_pred_constraint=tail_pred_constraint, apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/WN18RR/OMultBatch', data_path="%s/" % kg_path,
                          model_name='OMultBatch', tail_pred_constraint=tail_pred_constraint,
                          apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/WN18RR/ConvQBatch', data_path="%s/" % kg_path,
                          model_name='ConvQBatch', tail_pred_constraint=tail_pred_constraint,
                          apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/WN18RR/ConvOBatch', data_path="%s/" % kg_path,
                          model_name='ConvOBatch', tail_pred_constraint=tail_pred_constraint,
                          apply_range_constraint=apply_range_constraint)
    if show_ensembles:
        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/QMult', model_name='QMult'),
                           Reproduce().load_model(model_path='PretrainedModels/WN18RR/OMultBatch',
                                                  model_name='OMultBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)
        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/QMult', model_name='QMult'),
                           Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvQBatch',
                                                  model_name='ConvQBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/QMult', model_name='QMult'),
                           Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvOBatch',
                                                  model_name='ConvOBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/OMult', model_name='OMult'),
                           Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvQBatch',
                                                  model_name='ConvQBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(
                Reproduce().load_model(model_path='PretrainedModels/WN18RR/OMultBatch', model_name='OMultBatch'),
                Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvOBatch',
                                       model_name='ConvOBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(
                Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvQBatch', model_name='ConvQBatch'),
                Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConvOBatch', model_name='ConvOBatch'),
                Reproduce().load_model(model_path='PretrainedModels/WN18RR/OMultBatch',
                                       model_name='OMultBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))

if run_FB15K:
    kg_path = 'KGs/FB15k'
    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))
    Reproduce().reproduce(model_path='PretrainedModels/FB15K/QMult', data_path="%s/" % kg_path, model_name='QMultBatch',
                          tail_pred_constraint=tail_pred_constraint, apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/FB15K/OMult', data_path="%s/" % kg_path, model_name='OMultBatch',
                          tail_pred_constraint=tail_pred_constraint, apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/FB15K/ConvQ', data_path="%s/" % kg_path, model_name='ConvQBatch',
                          tail_pred_constraint=tail_pred_constraint, apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/FB15K/ConvO', data_path="%s/" % kg_path, model_name='ConvOBatch',
                          tail_pred_constraint=tail_pred_constraint, apply_range_constraint=apply_range_constraint)

if run_WN18:
    kg_path = 'KGs/WN18'
    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))
    Reproduce().reproduce(model_path='PretrainedModels/WN18/QMult', data_path="%s/" % kg_path, model_name='QMultBatch',
                          tail_pred_constraint=tail_pred_constraint, apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/WN18/ConvQ', data_path="%s/" % kg_path, model_name='ConvQBatch',
                          tail_pred_constraint=tail_pred_constraint, apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/WN18/OMult', data_path="%s/" % kg_path, model_name='OMultBatch',
                          tail_pred_constraint=tail_pred_constraint, apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/WN18/ConvO', data_path="%s/" % kg_path, model_name='ConvOBatch',
                          tail_pred_constraint=tail_pred_constraint, apply_range_constraint=apply_range_constraint)

if run_UMLS:
    kg_path = 'KGs/UMLS'
    print('###########################################{0}##################################################'.format(
        kg_path))
    Reproduce().reproduce(model_path='PretrainedModels/UMLS/QMult', data_path="%s/" % kg_path, model_name='QMultBatch',
                          tail_pred_constraint=tail_pred_constraint, apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/UMLS/ConvQ', data_path="%s/" % kg_path, model_name='ConvQBatch',
                          tail_pred_constraint=tail_pred_constraint, apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/UMLS/OMult', data_path="%s/" % kg_path, model_name='OMultBatch',
                          tail_pred_constraint=tail_pred_constraint, apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/UMLS/ConvO', data_path="%s/" % kg_path, model_name='ConvOBatch',
                          tail_pred_constraint=tail_pred_constraint, apply_range_constraint=apply_range_constraint)
    if show_ensembles:
        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/UMLS/ConvQ', model_name='ConvQBatch'),
                           Reproduce().load_model(model_path='PretrainedModels/UMLS/ConvO', model_name='ConvOBatch'),
                           Reproduce().load_model(model_path='PretrainedModels/UMLS/OMult', model_name='OMultBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/UMLS/ConvQ', model_name='ConvQBatch'),
                           Reproduce().load_model(model_path='PretrainedModels/UMLS/ConvO', model_name='ConvOBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/UMLS/QMult', model_name='QMultBatch'),
                           Reproduce().load_model(model_path='PretrainedModels/UMLS/OMult', model_name='OMultBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/UMLS/QMult', model_name='QMultBatch'),
                           Reproduce().load_model(model_path='PretrainedModels/UMLS/ConvQ', model_name='ConvQBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/UMLS/QMult', model_name='QMultBatch'),
                           Reproduce().load_model(model_path='PretrainedModels/UMLS/ConvO', model_name='ConvOBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/UMLS/OMult', model_name='OMultBatch'),
                           Reproduce().load_model(model_path='PretrainedModels/UMLS/ConvQ', model_name='ConvQBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/UMLS/OMult', model_name='OMultBatch'),
                           Reproduce().load_model(model_path='PretrainedModels/UMLS/ConvO', model_name='ConvOBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)
    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))

if run_Kinship:
    kg_path = 'KGs/KINSHIP'
    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))

    Reproduce().reproduce(model_path='PretrainedModels/Kinship/QMult', data_path="%s/" % kg_path,
                          model_name='QMultBatch', tail_pred_constraint=tail_pred_constraint,
                          apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/Kinship/ConvQ', data_path="%s/" % kg_path,
                          model_name='ConvQBatch', tail_pred_constraint=tail_pred_constraint,
                          apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/Kinship/OMult', data_path="%s/" % kg_path,
                          model_name='OMultBatch', tail_pred_constraint=tail_pred_constraint,
                          apply_range_constraint=apply_range_constraint)
    Reproduce().reproduce(model_path='PretrainedModels/Kinship/ConvO', data_path="%s/" % kg_path,
                          model_name='ConvOBatch', tail_pred_constraint=tail_pred_constraint,
                          apply_range_constraint=apply_range_constraint)
    if show_ensembles:
        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/Kinship/ConvQ', model_name='ConvQBatch'),
                           Reproduce().load_model(model_path='PretrainedModels/Kinship/ConvO', model_name='ConvOBatch'),
                           Reproduce().load_model(model_path='PretrainedModels/Kinship/OMult',
                                                  model_name='OMultBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/Kinship/ConvQ', model_name='ConvQBatch'),
                           Reproduce().load_model(model_path='PretrainedModels/Kinship/ConvO',
                                                  model_name='ConvOBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/Kinship/QMult', model_name='QMultBatch'),
                           Reproduce().load_model(model_path='PretrainedModels/Kinship/OMult',
                                                  model_name='OMultBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/Kinship/QMult', model_name='QMultBatch'),
                           Reproduce().load_model(model_path='PretrainedModels/Kinship/ConvQ',
                                                  model_name='ConvQBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/Kinship/QMult', model_name='QMultBatch'),
                           Reproduce().load_model(model_path='PretrainedModels/Kinship/ConvO',
                                                  model_name='ConvOBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/Kinship/OMult', model_name='OMultBatch'),
                           Reproduce().load_model(model_path='PretrainedModels/Kinship/ConvQ',
                                                  model_name='ConvQBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)

        Reproduce().reproduce_ensemble(
            model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/Kinship/OMult', model_name='OMultBatch'),
                           Reproduce().load_model(model_path='PretrainedModels/Kinship/ConvO',
                                                  model_name='ConvOBatch')),
            data_path="%s/" % kg_path, tail_pred_constraint=tail_pred_constraint)
    print(
        '###########################################     {0}     ##################################################'.format(
            kg_path))

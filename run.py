import os
import argparse
import sys
import pickle
from untils import extract_file
def read_args():
    parser = argparse.ArgumentParser()
     # Training our model
    parser.add_argument('-train', action='store_true', help='training model')
    parser.add_argument('-test', action='store_true', help='training model')
    parser.add_argument('-lora_RoBERTa', action='store_true', help='training model')
    parser.add_argument('-p_tuning_model', action='store_true', help='training model')
    parser.add_argument('-original', action='store_true', help='training model')
    parser.add_argument('-token_cls',action='store_true',  help='load model')
    parser.add_argument('-bart_model',action='store_true',  help='load model')
    parser.add_argument('-plbart_model',action='store_true',  help='load model')
    parser.add_argument('-gpt2_model',action='store_true',  help='load model')
    parser.add_argument('-codegpt_model',action='store_true',  help='load model')
    parser.add_argument('-shot_num', type=int, default=-1, help='the dimension of embedding vector')
    return parser
def p_tuning_model(params):
    if params.train:
        if params.original:
            print("train p-tuning model")
            train_cmd = "CUDA_VISIBLE_DEVICES=0 python p_tuning.py -train -train_data ../data/{}_data/{}_train.pkl -save-dir p_tuning_snapshot/original/{}"
            projects = ['qt','openstack']
            for project in projects:
                cmd = train_cmd.format(project, project, project)
                print(cmd)
                os.system(cmd)
        else:
            print("train p-tuning model")
            train_cmd = "CUDA_VISIBLE_DEVICES=0 python p_tuning.py -train -train_data ../config_dataset/data/{}/cc2vec/{}_train.pkl -save-dir p_tuning_snapshot/{}"
            projects = ['qt','openstack', 'jdt', 'platform', 'gerrit', 'go']
            projects = ['go']
            for project in projects:
                cmd = train_cmd.format(project, project, project)
                print(cmd)
                os.system(cmd)

        # print(train_cmd)
        # os.system(train_cmd)
    if params.test:
        if params.original:
            print("test p_tuning model")
            projects = ['qt', 'openstack']
            test_cmd = "CUDA_VISIBLE_DEVICES=0 python p_tuning.py -predict -pred_data ../data/{}_data/{}_test.pkl -load_model p_tuning_snapshot/original/{}/{}"
            for project in projects:
                pathlist = extract_file('p_tuning_snapshot/original/{}/'.format(project))
                pathlist = sorted(pathlist[:3])
                for path in pathlist:
                    cmd = test_cmd.format(project, project, project, path)
                    print(cmd)
                    os.system(cmd)
        else:
            print("test p_tuning model")
            projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
            # projects=['qt']
            test_cmd = "CUDA_VISIBLE_DEVICES=0 python p_tuning.py -predict -pred_data ../config_dataset/data/{}/cc2vec/{}_test.pkl -load_model p_tuning_snapshot/{}/{}"
            for project in projects:
                pathlist = extract_file('p_tuning_snapshot/{}/'.format(project))
                pathlist = sorted(pathlist[:3])
                for path in pathlist:
                    cmd = test_cmd.format(project, project, project, path)
                    print(cmd)
                    os.system(cmd)

            # print(pathlist)
        # print(test_cmd)
        # os.system(test_cmd)
def bart_model(params):
    if params.train:
        if params.original:
            print("train bart-base model")
            train_cmd = "CUDA_VISIBLE_DEVICES=2 python plbart.py -train -train_data ../data/{}_data/{}_train.pkl -save-dir bart_snapshot/original/{} -dictionary_data ../data/{}_dict.pkl"
            projects = ['qt','openstack']
            for project in projects:
                cmd = train_cmd.format(project, project, project,project)
                print(cmd)
                os.system(cmd)
    if params.test:
        if params.original:
            print("test bart-base model")
            projects = ['qt', 'openstack']
            test_cmd = "CUDA_VISIBLE_DEVICES=0 python plbart.py -predict -pred_data ../data/{}_data/{}_test.pkl -load_model bart_snapshot/original/{}/{} -dictionary_data ../data/{}_dict.pkl -weight"
            for project in projects:
                pathlist = extract_file('bart_snapshot/original/{}/'.format(project))[:3]

                for path in pathlist:
                    cmd = test_cmd.format(project, project, project, path,project)
                    print(cmd)
                    os.system(cmd)
def plbart_model(params):
    if params.train:
        if params.original:
            print("train plbart-base model")
            train_cmd = "CUDA_VISIBLE_DEVICES=0 python bart.py -train -train_data ../data/{}_data/{}_train.pkl -save-dir plbart_snapshot/original/{} -dictionary_data ../data/{}_dict.pkl"
            projects = ['openstack']
            for project in projects:
                cmd = train_cmd.format(project, project, project,project)
                print(cmd)
                os.system(cmd)
    if params.test:
        if params.original:
            print("test plbart-base model")
            projects = ['qt', 'openstack']
            test_cmd = "CUDA_VISIBLE_DEVICES=0 python bart.py -predict -pred_data ../data/{}_data/{}_test.pkl -load_model plbart_snapshot/original/{}/{} -dictionary_data ../data/{}_dict.pkl -weight"
            for project in projects:
                pathlist = extract_file('plbart_snapshot/original/{}/'.format(project))
                pathlist = sorted(pathlist[:3])
                for path in pathlist:
                    cmd = test_cmd.format(project, project, project, path,project)
                    print(cmd)
                    os.system(cmd)
def gpt2_model(params):
    if params.train:
        if params.original:
            print("train plbart-base model")
            train_cmd = "CUDA_VISIBLE_DEVICES=0 python GPT2.py -train -train_data ../data/{}_data/{}_train.pkl -save-dir gpt2_snapshot/original/{} -dictionary_data ../data/{}_dict.pkl"
            projects = ['qt','openstack']
            for project in projects:
                cmd = train_cmd.format(project, project, project,project)
                print(cmd)
                os.system(cmd)
    if params.test:
        if params.original:
            print("test plbart-base model")
            projects = ['qt', 'openstack']
            test_cmd = "CUDA_VISIBLE_DEVICES=0 python GPT2.py -predict -pred_data ../data/{}_data/{}_test.pkl -load_model gpt2_snapshot/original/{}/{} -dictionary_data ../data/{}_dict.pkl -weight"
            for project in projects:
                pathlist = extract_file('gpt2_snapshot/original/{}/'.format(project))
                pathlist = sorted(pathlist[:3])
                for path in pathlist:
                    cmd = test_cmd.format(project, project, project, path,project)
                    print(cmd)
                    os.system(cmd)
def codegpt_model(params):
    if params.train:
        if params.original:
            print("train plbart-base model")
            train_cmd = "CUDA_VISIBLE_DEVICES=0 python CodeGPT.py -train -train_data ../data/{}_data/{}_train.pkl -save-dir codegpt_snapshot/original/{} -dictionary_data ../data/{}_dict.pkl -dictionary_data ../data/{}_dict.pkl"
            projects = ['qt','openstack']
            for project in projects:
                cmd = train_cmd.format(project, project, project,project,project)
                print(cmd)
                os.system(cmd)
    if params.test:
        if params.original:
            print("test plbart-base model")
            projects = ['qt', 'openstack']
            test_cmd = "CUDA_VISIBLE_DEVICES=0 python CodeGPT.py -predict -pred_data ../data/{}_data/{}_test.pkl -load_model codegpt_snapshot/original/{}/{} -dictionary_data ../data/{}_dict.pkl -weight"
            for project in projects:
                pathlist = extract_file('plbart_snapshot/original/{}/'.format(project))
                pathlist = sorted(pathlist[:3])
                for path in pathlist:
                    cmd = test_cmd.format(project, project, project, path,project)
                    print(cmd)
                    os.system(cmd)
def lora_RoBERTa_model(params):
    if params.train:
        if params.original:
            print("train LoRA-large model")
            train_cmd = "CUDA_VISIBLE_DEVICES=0 python def_read_file.py -train -train_data ../data/{}_data/{}_train.pkl -save-dir lora_msnapshot/original/large_{}"
            projects = ['qt','openstack']
            for project in projects:
                cmd = train_cmd.format(project, project, project)
                print(cmd)
                os.system(cmd)
        elif params.token_cls:
            print("train LoRA-base model")
            train_cmd = "CUDA_VISIBLE_DEVICES=2 python def_read_file.py -train -train_data precessed/{}/train.pkl -save-dir pre_snapshot/{}"
            projects = ['qt']
            for project in projects:
                cmd = train_cmd.format(project, project)
                print(cmd)
                os.system(cmd)
        else:
            print("train LoRA-CodeBERT model")
            train_cmd = "CUDA_VISIBLE_DEVICES=2 python def_read_file.py -train -train_data ../config_dataset/data/{}/cc2vec/{}_train.pkl -save-dir lora_msnapshot/{}"
            projects = ['qt','openstack', 'jdt', 'platform', 'gerrit', 'go']
            for project in projects:
                cmd = train_cmd.format(project, project, project)
                print(cmd)
                os.system(cmd)

        # print(train_cmd)
        # os.system(train_cmd)
    if params.test:
        if params.original:
            print("test LoRA-base model")
            projects = ['qt', 'openstack']
            test_cmd = "CUDA_VISIBLE_DEVICES=2 python def_read_file.py -predict -pred_data ../data/{}_data/{}_test.pkl -load_model lora_msnapshot/original/roberta_{}/{}"
            for project in projects:
                pathlist = extract_file('lora_msnapshot/original/roberta_{}/'.format(project))
                pathlist = sorted(pathlist[:3])
                for path in pathlist:
                    cmd = test_cmd.format(project, project, project, path)
                    print(cmd)
                    os.system(cmd)
        elif params.token_cls:
            print("test LoRA-base model")
            test_cmd = "CUDA_VISIBLE_DEVICES=2 python def_read_file.py  -predict -pred_data precessed/{}/test.pkl -load_model pre_snapshot/{}/{}"
            projects = ['qt']
            for project in projects:
                pathlist = extract_file('pre_snapshot/{}/'.format(project))
                pathlist = sorted(pathlist[:3])
                for path in pathlist:
                    cmd = test_cmd.format(project, project, path)
                    print(cmd)
                    os.system(cmd)

        else:
            print("test LoRA-CodeBERT model")
            projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
            test_cmd = "CUDA_VISIBLE_DEVICES=2 python def_read_file.py -predict -pred_data ../config_dataset/data/{}/cc2vec/{}_test.pkl -load_model lora_msnapshot/{}/{} -weight"
            for project in projects:
                pathlist = extract_file('pre_snapshot/{}/'.format(project))
                pathlist = sorted(pathlist[:3])
                for path in pathlist:
                    cmd = test_cmd.format(project, project, project, path)
                    print(cmd)
                    os.system(cmd)

            # print(pathlist)
        # print(test_cmd)
        # os.system(test_cmd)
def roberta_model(params):
    if params.train:
        if params.original:
            print("train roberta-base model")
            train_cmd = "CUDA_VISIBLE_DEVICES=0 python RoBERTa.py -train -train_data ../data/{}_data/{}_train.pkl -save-dir roberta_snapshot/original/{} -dictionary_data ../data/{}_dict.pkl"
            projects = ['qt','openstack']
            for project in projects:
                cmd = train_cmd.format(project, project, project,project)
                print(cmd)
                os.system(cmd)
    if params.test:
        if params.original:
            print("test roberta-base model")
            projects = ['qt', 'openstack']
            test_cmd = "CUDA_VISIBLE_DEVICES=0 python RoBERTa.py -predict -pred_data ../data/{}_data/{}_test.pkl -load_model roberta_snapshot/original/{}/{} -dictionary_data ../data/{}_dict.pkl -weight"
            for project in projects:
                pathlist = extract_file('roberta_snapshot/original/{}/'.format(project))
                pathlist = sorted(pathlist[:3])
                for path in pathlist:
                    cmd = test_cmd.format(project, project, project, path,project)
                    print(cmd)
                    os.system(cmd)
def codebert_model(params):
    if params.train:
        if params.original:
            print("train plbart-base model")
            train_cmd = "CUDA_VISIBLE_DEVICES=0 python CodeBERT.py -train -train_data ../data/{}_data/{}_train.pkl -save-dir codebert_snapshot/original/{} -dictionary_data ../data/{}_dict.pkl -dictionary_data ../data/{}_dict.pkl"
            projects = ['qt','openstack']
            for project in projects:
                cmd = train_cmd.format(project, project, project,project,project)
                print(cmd)
                os.system(cmd)
    if params.test:
        if params.original:
            print("test plbart-base model")
            projects = ['qt','openstack']
            test_cmd = "CUDA_VISIBLE_DEVICES=0 python CodeBERT.py -predict -pred_data ../data/{}_data/{}_test.pkl -load_model codebert_snapshot/original/{}/{} -dictionary_data ../data/{}_dict.pkl -weight"
            for project in projects:
                pathlist = extract_file('codebert_snapshot/original/{}/'.format(project))
                pathlist = sorted(pathlist[:3])
                for path in pathlist:
                    cmd = test_cmd.format(project, project, project, path,project)
                    print(cmd)
                    os.system(cmd)
def zero_shot():
    print("test zero-shot model")
    projects = ['qt', 'openstack']
    python_file_list=['RoBERTa.py','CodeBERT.py','GPT2.py','CodeGPT.py','plbart.py','bart.py']

    test_cmd = "CUDA_VISIBLE_DEVICES=1 python {} -predict -pred_data ../data/{}_data/{}_test.pkl -zero_shot -dictionary_data ../data/{}_dict.pkl"
    for py_file in python_file_list:
        for project in projects:
            cmd = test_cmd.format(py_file,project, project,project)
            print(cmd)
            os.system(cmd)
from tqdm import tqdm
def few_shot(params):
    project=['qt','openstack']
    shot_num = [10, 100, 500, 1000, 2000]
    shot_num=[2400]
    python_file_list = ['RoBERTa.py', 'CodeBERT.py', 'GPT2.py', 'CodeGPT.py', 'plbart.py', 'bart.py']
    model_list = ['roberta', 'codebert', 'gpt2', 'codegpt',  'bart','plbart']
    if params.train:
        traim_cmd = "CUDA_VISIBLE_DEVICES=1 python {} -train -train_data ../data/few_shot_data/{}_shot/{}_train.pkl -save-dir few_shot_snapshot/{}_{}/{}/ -dictionary_data ../data/{}_dict.pkl"
        for index in range(len(python_file_list)):
            for p in project:
                for shot in shot_num:
                    print('train {}_{}_{} model'.format(p,shot,model_list[index]))
                    if shot<16:

                        cmd = traim_cmd.format(python_file_list[index],shot,p,model_list[index],shot,p,p)+ ' -batch_size 8'
                        print(cmd)
                        os.system(cmd)
                    else:
                        cmd = traim_cmd.format(python_file_list[index], shot, p, model_list[index], shot, p, p)
                        print(cmd)
                        os.system(cmd)

                    # os.system(cmd)
    if params.test:
        test_cmd = "CUDA_VISIBLE_DEVICES=0 python {} -predict -pred_data ../data/{}_data/{}_test.pkl -load_model few_shot_snapshot/{}_{}/{}/{} -dictionary_data ../data/{}_dict.pkl"
        for index in range(len(python_file_list)):
            for p in project:

                for shot in shot_num:
                    path_list = extract_file('few_shot_snapshot/{}_{}/{}'.format(model_list[index], shot, p))
                    path_list = sorted(path_list)

                    print('test {}_{}_{} model'.format(p, shot, model_list[index]))
                    for tmp_path in path_list:
                        if shot < 16:
                            cmd = test_cmd.format(python_file_list[index], p, p, model_list[index], shot, p, tmp_path,
                                                  p) + ' -batch_size 8'
                            print(cmd)
                            os.system(cmd)
                        else:
                            cmd = test_cmd.format(python_file_list[index], p, p, model_list[index], shot, p, tmp_path,
                                                  p)
                            print(cmd)
                            os.system(cmd)

if __name__ == '__main__':
    argparse = read_args()
    params = argparse.parse_args(sys.argv[1:])

    # if params.lora_RoBERTa:
    #     lora_RoBERTa_model(params)
    # elif params.p_tuning_model:
    #     p_tuning_model(params)
    # elif params.bart_model:
    #     bart_model(params)
    # elif params.plbart_model:
    #     plbart_model(params)
    # shot_num=[1,5,10,20,30,50,100,200,300,500,1000,2000]
    # tmp_shot='-shot_num '+str(shot_num[0])
    few_shot(params)
    # if params.train:
    #
    #     # print("train gpt2 model====================")
    #     # gpt2_model(params)
    #     # print("train codegpt model====================")
    #     # codegpt_model(params)
    #     #
    #     # print("train roberta model====================")
    #     # roberta_model(params)
    #     # print("train codebert model====================")
    #     # codebert_model(params)
    #     # few_shot(params)
    # elif params.test:
    #
    #     # print("test roberta model====================")
    #     # roberta_model(params)
    #     # # print("test codebert model====================")
    #     # # codebert_model(params)
    #     # #
    #     # print("test gpt2 model====================")
    #     # gpt2_model(params)
    #     # print("test codegpt model====================")
    #     # codegpt_model(params)
    #     #
    #     # # print("test bart-base model====================")
    #     # # bart_model(params)
    #     # print("test plbart-base model====================")
    #     # plbart_model(params)
    #     zero_shot()
    # else:
    #     print("please choose a model to train")
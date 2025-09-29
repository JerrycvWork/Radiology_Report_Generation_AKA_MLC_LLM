import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devicess = [0]
import argparse
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pandas as pd
import torch.nn.functional as F
from asyloss import AsymmetricLoss
import numpy as np
from dataset_rebuild import CustomDataset_classification_train_iuxray,CustomDataset_classification_test_iuxray,CustomDataset_classification_val_iuxray,CustomDataset_classification_train_mimic_cxr,CustomDataset_classification_test_mimic_cxr,CustomDataset_classification_val_mimic_cxr
from sklearn import metrics
import json
import numpy as np


class hp:

    train_or_test = 'train'
    dataset="iuxray"
    output_dir = 'Multi_Label_Classification_Session/sample_ckpt/convnext_iuxray/'
    aug = False
    latest_checkpoint_file = 'checkpoint_0001.pt'
    total_epochs = 1
    epochs_per_checkpoint = 1
    batch_size = 1
    ckpt = None
    init_lr = 0.0002
    scheduer_step_size = 20
    scheduer_gamma = 0.8
    debug = False
    mode = '2d' # '2d or '3d'
    eval_epoch=5

    net='mobilenet'

    iuxray_image_folder="IUXray/NLMCXR_png/"
    mimic_image_folder="mimic_cxr/"


def parse_training_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--Transformer', type=str, default=1, required=False,
                        help='Change the use of Optimizer')
    parser.add_argument('--train_or_test', type=str, default=hp.train_or_test, required=False,
                        help='Directory to save checkpoints')
    parser.add_argument('--net', type=str, default=hp.net, required=False,
                        help='Directory to save checkpoints')
    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False,
                        help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default=hp.latest_checkpoint_file,
                        help='Store the latest checkpoint in each epoch')
    parser.add_argument('--kfold', type=str, default=hp.latest_checkpoint_file,
                        help='Store the latest checkpoint in each epoch')
    parser.add_argument('--direction', type=str, default=hp.latest_checkpoint_file,
                        help='Store the latest checkpoint in each epoch')
    parser.add_argument('--dataset', type=str, default="iuxray",
                        help='Store the latest checkpoint in each epoch')
    parser.add_argument('--test_ckpt_dir', type=str, default=hp.output_dir, required=False,
                        help='Directory to save checkpoints')
    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=hp.total_epochs, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=hp.epochs_per_checkpoint,
                          help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=hp.batch_size, help='batch-size')
    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=hp.ckpt,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--init-lr", type=float, default=hp.init_lr, help="learning rate")
    # TODO
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true',
                          help='disable uniform initialization of batchnorm layer weight')



    return parser


def train(run_cluster=1,run_level=1,run_class_num=19,train_csv_path="",val_csv_path=""):
    parser = argparse.ArgumentParser(description='Multi_Label Classification')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()
    # print(args)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    os.makedirs(args.output_dir, exist_ok=True)

    print(args.net)

    if hp.mode == '2d':

        if args.net == 'resnet':
            print("Load: ", "ResNet")
            from Multi_Label_Classification_Session.sample_model_structure.resnet import resnet50
            model = resnet50(num_classes=int(run_class_num))

        if args.net == 'resnext':
            print("Load: ", "ResNext")
            from Multi_Label_Classification_Session.sample_model_structure.resnext import resnext50
            model = resnext50(class_names=int(run_class_num))

        if args.net == 'vgg16':
            print("Load: ", "VGG16")
            from timm.models.vgg import vgg16_bn, vgg16
            model = vgg16_bn(num_classes=int(run_class_num), in_chans=1, pretrained=True)

        if args.net == 'convnext':   # Valid
            print("Load: ", "ConvNext")
            from timm.models.convnext import convnext_base
            model = convnext_base(num_classes=int(run_class_num), in_chans=1, pretrained=True)
        
        else:
            print("Network structure not implement yet.")


    #model = torch.nn.DataParallel(model, device_ids=0)

    print("Adam Optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=hp.scheduer_step_size, gamma=hp.scheduer_gamma)


    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file),
                          map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0

    model.cuda()
    writer = SummaryWriter(args.output_dir)

    ## Rewrite
    print(run_class_num)
    print(run_level)

    if args.dataset=="iuxray":
        train_dataset = CustomDataset_classification_train_iuxray(classes=run_class_num, level=run_level, train_csv=train_csv_path,image_folder=hp.iuxray_image_folder)
    elif args.dataset=="mimic":
        train_dataset = CustomDataset_classification_train_mimic_cxr(classes=run_class_num, level=run_level,train_csv=train_csv_path,image_folder=hp.mimic_image_folder)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

    model.train()

    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)

    for epoch in range(1, epochs + 1):
        print("epoch:" + str(epoch))
        epoch += elapsed_epochs

        num_iters = 0
        for i, (image, label) in enumerate(train_loader):

            print(f"Batch: {i}/{len(train_loader)} epoch {epoch}")

            optimizer.zero_grad()

            x = image
            target = label

            x = x.type(torch.FloatTensor).cuda()
            target = target.cuda()  # (batch,3,num_classes)

            outputs = model(x)

            ## Multiple_bit_loss
            loss=criterion(outputs,target)

            num_iters += 1
            loss.backward()

            optimizer.step()
            iteration += 1

            print("loss:" + str(loss.item()))
            writer.add_scalar('Training/Loss', loss.item(), iteration)

            break

        scheduler.step()

        # Store latest checkpoint in each epoch
        os.makedirs(args.output_dir+"/"+str(run_level),exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,

            },
            os.path.join(args.output_dir,str(run_level), args.latest_checkpoint_file),
        )

        # Save checkpoint
        if epoch % args.epochs_per_checkpoint == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(args.output_dir,str(run_level), f"checkpoint_{epoch:04d}.pt"),
            )
        
        # Validation
        if epoch % hp.eval_epoch == 0:
            val(run_cluster,run_level,run_class_num,val_csv_path,model)

    writer.close()


def test(run_cluster=1,run_level=1,run_class_num=19,test_csv_path="",test_ckpt=""):
    parser = argparse.ArgumentParser(description='Multi_Label Classification')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if hp.mode == '2d':

        if args.net == 'resnet':  # Valid
            print("Load: ", "ResNet")
            from Multi_Label_Classification_Session.sample_model_structure.resnet import resnet50
            model = resnet50(num_classes=int(run_class_num))

        if args.net == 'resnext':
            print("Load: ", "ResNext")
            from Multi_Label_Classification_Session.sample_model_structure.resnext import resnext50
            model = resnext50(class_names=int(run_class_num))

        if args.net == 'vgg16':
            print("Load: ", "VGG16")
            from timm.models.vgg import vgg16_bn, vgg16
            model = vgg16_bn(num_classes=int(run_class_num), in_chans=1, pretrained=True)

        if args.net == 'convnext':
            print("Load: ", "ConvNext")
            from timm.models.convnext import convnext_base
            model = convnext_base(num_classes=int(run_class_num), in_chans=1, pretrained=True)
                
        else:
            print("Network structure not implement yet.")


    #model = torch.nn.DataParallel(model, device_ids=devicess, output_device=[1])

    print("load model:")
    print(os.path.join(test_ckpt,str(run_level),args.latest_checkpoint_file))
    ckpt = torch.load(os.path.join(test_ckpt,str(run_level), args.latest_checkpoint_file),
                      map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["model"])
    model.cuda()

    if args.dataset == "iuxray":
        test_dataset = CustomDataset_classification_test_iuxray(classes=run_class_num, level=run_level,
                                                           test_csv=test_csv_path,image_folder=hp.iuxray_image_folder)
    elif args.dataset == "mimic":
        test_dataset = CustomDataset_classification_test_mimic_cxr(classes=run_class_num, level=run_level,
                                                                     test_csv=test_csv_path,image_folder=hp.mimic_image_folder)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=True,
                             pin_memory=True,
                             drop_last=True)

    model.eval()

    f = open(os.path.join(test_ckpt,str(run_level)) + "/test_result.txt", 'w')
    tt = []
    ct = 0
    f_properties = ["filename", "predict_score", "predict_result", "gt"]
    for i, (image, label, name) in enumerate(test_loader):
        tt.append([])
        x = image
        y = label
        file_name = name

        x = x.type(torch.FloatTensor).cuda()
        outputs = model(x)

        print(file_name)
        f.write(file_name[0])
        f.write('\n')
        f.write("Predict Score: ")
        f.write(str(F.sigmoid(outputs).cpu().detach().numpy()))
        f.write("Real Predicts: ")
        temp_score = F.sigmoid(outputs).cpu().detach().numpy()
        temp_score[temp_score > 0.5] = 1
        temp_score[temp_score <= 0.5] = 0
        print(temp_score)
        f.write(str(temp_score))
        f.write('\n')
        f.write("gts: ")
        f.write(str(y))
        f.write('\n')

        tt[ct].append(file_name[0])
        tt[ct].append(str(F.sigmoid(outputs).cpu().detach().numpy()))
        tt[ct].append(str(temp_score))
        tt[ct].append(str(y))

        ct += 1

    final = pd.DataFrame(columns=f_properties, data=tt)
    final.to_csv(os.path.join(test_ckpt,str(run_level)) + "/test_result.csv",index=False)



def val(run_cluster=1,run_level=1,run_class_num=19,val_csv_path="",val_model=None):

    validation_model=val_model
    parser = argparse.ArgumentParser(description='Multi_Label Classification')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if args.dataset == "iuxray":
        test_dataset = CustomDataset_classification_val_iuxray(classes=run_class_num, level=run_level,
                                                           test_csv=val_csv_path,image_folder=hp.iuxray_image_folder)
    elif args.dataset == "mimic":
        test_dataset = CustomDataset_classification_val_mimic_cxr(classes=run_class_num, level=run_level,
                                                                     test_csv=val_csv_path,image_folder=hp.mimic_image_folder)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=True,
                             pin_memory=True,
                             drop_last=True)

    validation_model.eval()

    f = open(os.path.join(args.output_dir,str(run_level)) + "/val_result.txt", 'w')
    tt = []
    ct = 0
    f_properties = ["filename", "predict_score", "predict_result", "gt"]
    for i, (image, label, name) in enumerate(test_loader):
        tt.append([])
        x = image
        y = label
        file_name = name

        x = x.type(torch.FloatTensor).cuda()
        outputs = model(x)

        print(file_name)
        f.write(file_name[0])
        f.write('\n')
        f.write("Predict Score: ")
        f.write(str(F.sigmoid(outputs).cpu().detach().numpy()))
        f.write("Real Predicts: ")
        temp_score = F.sigmoid(outputs).cpu().detach().numpy()
        temp_score[temp_score > 0.5] = 1
        temp_score[temp_score <= 0.5] = 0
        print(temp_score)
        f.write(str(temp_score))
        f.write('\n')
        f.write("gts: ")
        f.write(str(y))
        f.write('\n')

        tt[ct].append(file_name[0])
        tt[ct].append(str(F.sigmoid(outputs).cpu().detach().numpy()))
        tt[ct].append(str(temp_score))
        tt[ct].append(str(y))

        ct += 1

    final = pd.DataFrame(columns=f_properties, data=tt)
    final.to_csv(os.path.join(args.output_dir,str(run_level)) + "/val_result.csv",index=False)

    ## (Temporary Function for print the validation performance)
    val_performance_print(os.path.join(args.output_dir,str(run_level)) + "/val_result.csv")


def val_performance_print(result_csv_path):

    result_csv=pd.read_csv(result_csv_path)

    Total_TP = 0
    Total_TN = 0
    Total_FP = 0
    Total_FN = 0

    Total_Acc = []
    Total_Sen = []  # TPR
    Total_Spe = []  # TNR
    Total_F1 = []
    Total_MCC = []

    Total_PPV = []
    Total_NPV = []

    f_name = []


    result_csv['accuracy'] = [''] * len(result_csv)
    result_csv['TPR'] = [''] * len(result_csv)
    result_csv['TNR'] = [''] * len(result_csv)
    result_csv['f1_score'] = [''] * len(result_csv)
    result_csv['mcc'] = [''] * len(result_csv)

    result_csv['PPV'] = [''] * len(result_csv)
    result_csv['NPV'] = [''] * len(result_csv)

    for s1 in range(len(result_csv)):
            result_str = result_csv['predict_result'][s1]
            gt_str = result_csv['gt'][s1]

            result_numpy = np.fromstring(result_str, sep=",")
            # print(s1)
            gt_numpy = np.fromstring(gt_str, sep=",")

            if np.sum(gt_numpy) >= 0:
                c_m = metrics.confusion_matrix(gt_numpy, result_numpy)
                f1_score = metrics.f1_score(gt_numpy, result_numpy)
                mcc = metrics.matthews_corrcoef(gt_numpy, result_numpy)
                accuracy = metrics.accuracy_score(gt_numpy, result_numpy)

                if c_m.shape[0] > 1:
                    TP = c_m[1][1]
                    FN = c_m[1][0]
                    FP = c_m[0][1]
                    TN = c_m[0][0]
                elif c_m.shape[0] == 1:
                    TP = c_m[0][0]
                    FN = 0
                    FP = 0
                    TN = 0

                TPR = (TP) / (TP + FN)

                if TN > 0:
                    TNR = (TN) / (TN + FP)
                elif TN == 0 and FP > 0:
                    TNR = 0
                elif TN == 0 and FP == 0:
                    TNR = 1

                if TP == 0 and FP == 0:
                    PPV = 0
                else:
                    PPV = (TP) / (TP + FP)

                if TN > 0:
                    NPV = (TN) / (TN + FN)
                elif TN == 0 and FN > 0:
                    NPV = 0
                elif TN == 0 and FN == 0:
                    NPV = 0

                result_csv['accuracy'][s1] = accuracy
                result_csv['TPR'][s1] = TPR
                result_csv['TNR'][s1] = TNR
                result_csv['f1_score'][s1] = f1_score
                result_csv['mcc'][s1] = mcc

                result_csv['PPV'][s1] = PPV
                result_csv['NPV'][s1] = NPV

    print("Performance Summary")

    print("mean")
    print("Accuracy  ", np.mean(result_csv['accuracy']))
    print("Sensitivity  ", np.mean(result_csv['TPR']))
    print("Specificity  ", np.mean(result_csv['TNR']))
    print("F1-Score  ", np.mean(result_csv['f1_score']))
    print("MCC  ", np.mean(result_csv['mcc']))

    print("std")
    print("Accuracy  ", np.std(result_csv['accuracy']))
    print("Sensitivity  ", np.std(result_csv['TPR']))
    print("Specificity  ", np.std(result_csv['TNR']))
    print("F1-Score  ", np.std(result_csv['f1_score']))
    print("MCC  ", np.std(result_csv['mcc']))

    result_csv.to_csv(result_csv_path.replace('.csv',"_with_performance.csv"),index=False)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Multi-Label Classification')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    with open(
            "Multi_Label_Classification_Session/" + args.dataset + "_setting.json",
            "r") as f:
        data = json.load(f)

    train_csv_path=data["Train_csv_datapath"]
    test_csv_path=data["Test_csv_datapath"]
    val_csv_path = data["Val_csv_datapath"]
    cluster_num = data["Keyword_Cluster_number"]
    full_cluster_level = data["Corresponding_Cluster_Frequency"]
    full_cluster_classes_num = data["Corresponding_Cluster_keyword_Number"]

    print(cluster_num)
    print(full_cluster_level)
    print(full_cluster_classes_num)

    if args.train_or_test == 'train':
        for i in range(cluster_num):
           train(i,full_cluster_level[i],full_cluster_classes_num[i],train_csv_path,val_csv_path)
    elif args.train_or_test == 'test':
        for i in range(cluster_num):
            test(i, full_cluster_level[i], full_cluster_classes_num[i], test_csv_path, args.test_ckpt_dir)

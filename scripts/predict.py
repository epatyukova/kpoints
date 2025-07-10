from datamodule import GNNDataModule
from utils import load_yaml_config
from modelmodule import GNNModel
from pytorch_lightning import Trainer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
import torch
import os
from sklearn.preprocessing import RobustScaler

sys.path.append(str(Path(__file__).resolve().parent))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction script")

    parser.add_argument("--config_file",
                        default="cgcnn_config.yaml",
                        help="Provide the experiment configuration file")
    parser.add_argument("--checkpoint_path",
                        default="cgcnn_models/cgcnn_trained_models/july_3_magpie_compound/",
                        help="Provide the path to model checkpoint")
    parser.add_argument("--output_name",
                        default="output/cgcnn_july_magpie_compound.csv",
                        help="Provide the path to save prediction results")

    args = parser.parse_args(sys.argv[1:])
    config = load_yaml_config(args.config_file)
    
    config['data']['lmdb_exist']=True
    try:
        data = GNNDataModule(**config['data'])
    except:
        config['data']['lmdb_exist']=False
        data = GNNDataModule(**config['data'])
    model = GNNModel(**config)

    list_of_checkpoints = os.listdir(args.checkpoint_path)
    list_of_checkpoints = [f for f in list_of_checkpoints if f != ".DS_Store"]
    
    df=pd.DataFrame()
    for i,checkpoint in enumerate(list_of_checkpoints):
        trainer = Trainer(max_epochs=5,accelerator='cpu', devices=1)
        pred = trainer.predict(model, ckpt_path=os.path.join(args.checkpoint_path,checkpoint), datamodule=data)
    
        truth=[]
        prediction=[]
        if config['model']['robust_regression']:
            logstd=[]
        elif config['model']['classification']:
            probs=[]
        test_ids=[]

        for idx in range(len(pred)):
            if config['model']['robust_regression']:
                prediction.append(pred[idx][0])
                truth.append(pred[idx][2])
                logstd.append(pred[idx][1])
                test_ids+=pred[idx][3]
            elif config['model']['classification']:
                prediction.append(pred[idx][0])
                truth.append(pred[idx][2])
                probs.append(pred[idx][1])
                test_ids+=pred[idx][3]
            else:
                prediction.append(pred[idx][0])
                truth.append(pred[idx][1])
                test_ids+=pred[idx][2]

        if config['model']['robust_regression']:        
            truth=torch.cat(truth,dim=0).squeeze(-1)
            prediction=torch.cat(prediction,dim=0).squeeze(-1)
            logstd=torch.cat(logstd,dim=0).squeeze(-1)
        elif config['model']['classification']:        
            truth=torch.cat(truth,dim=0).squeeze(-1)
            prediction=torch.cat(prediction,dim=0).squeeze(-1)
            probs=torch.cat(probs,dim=0).squeeze(-1)
        else:
            truth=torch.cat(truth,dim=0).squeeze(-1)
            prediction=torch.cat(prediction,dim=0).squeeze(-1)

        df['id']=np.array(test_ids)
        df['truth_scaled']=truth
        df['prediction'+str(i)]=prediction
        if config['model']['robust_regression']:
            df['logstd'+str(i)]=logstd

    num=len(list_of_checkpoints)
    prediction_list=[]
    for i in range(num):
        prediction_list.append('prediction'+str(i))

    df["avg_prediction_scaled"] = df[prediction_list].mean(axis=1)
    truth = truth
    prediction = df["avg_prediction_scaled"].values
    
    if config['data']['scale_y']:
        scaler = RobustScaler()
        data = pd.read_csv(os.path.join(config['data']['root_dir'],config['data']['id_prop_csv']),header=None)
        y=np.array(data[1].values).reshape(-1, 1)
        scaler = scaler.fit(y)
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(len(prediction))
        df['prediction'] = prediction
        truth = scaler.inverse_transform(truth.reshape(-1, 1)).reshape(len(truth))
        df['truth'] = truth
    else:
        df['prediction'] = prediction
        df['truth'] = truth

    df.to_csv(args.output_name)

    # metrics
    if config['model']['classification']:
        
        acc = accuracy_score(truth.cpu().numpy(), prediction)
        f1 = f1_score(truth.cpu().numpy(), prediction, average='macro')
        mcc = matthews_corrcoef(truth.cpu().numpy(), prediction)
        cm = confusion_matrix(truth.cpu().numpy(), prediction)
        

        print(f'Model name: {config["model"]["name"]}')
        print(f'Test set acc: {acc}')
        print(f'Test set f1_score: {f1}')
        print(f'Test set mcc: {mcc}')
        print(f'Test set confusion matrix: {cm}')
    else:
        mse = mean_squared_error(truth,prediction)
        mae = mean_absolute_error(truth,prediction)
        mape = mean_absolute_percentage_error(truth,prediction)
        r2 = r2_score(truth,prediction)
        
        print(f'Model name: {config["model"]["name"]}')
        print(f'Test set MAE: {mae}')
        print(f'Test set MAPE: {mape}')
        print(f'Test set MSE: {mse}')
        print(f'Test set R2 score: {r2}')




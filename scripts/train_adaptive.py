
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

def split(input,history,state):
    train_input=input[:,:history*state]
    train_info=input[:,history*state:]
    return train_input, train_info

class AdaptationModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers=[]
        layers_mlp=[]
        layers_mlp.append(nn.LazyLinear(56))
        layers_mlp.append(nn.ELU())
        layers_mlp.append(nn.LazyLinear(128))
        layers_mlp.append(nn.ELU())
        layers_mlp.append(nn.LazyLinear(128))
        self.mlp=nn.Sequential(*layers_mlp)
        layers.append(nn.Conv1d(128,128,8,stride=3))
        layers.append(nn.ELU())
        layers.append(nn.Conv1d(128,128,5,stride=1))
        layers.append(nn.ELU())
        layers.append(nn.Conv1d(128,128,5,stride=1))
        self.adapt_core=nn.Sequential(*layers)
        self.linear1=nn.LazyLinear(256)
        self.nonlinear1=nn.ELU()
        self.linear2=nn.Linear(256,128)
        self.nonlinear2=nn.ELU()
        self.linear3=nn.Linear(128,14)

    def forward(self, history):
        x=self.mlp(history)
        x=x.permute(0,2,1)
        x=self.adapt_core(x)
        x=x.reshape(-1,x.shape[1]*x.shape[2])
        x=self.linear1(x)
        x=self.nonlinear1(x)
        x=self.linear2(x)
        x=self.nonlinear2(x)
        x=self.linear3(x)
        return x

class Info_Encoder(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
        layers=[]
        layers.append(nn.Linear(in_features=14, out_features=360, bias=False))
        layers.append(nn.ELU(alpha=1.0))
        layers.append(nn.Linear(in_features=360, out_features=256, bias=False))
        layers.append(nn.ELU(alpha=1.0))
        layers.append(nn.Linear(in_features=256, out_features=128, bias=False))
        self.layers=nn.Sequential(*layers)
    
    def forward(self,input):
        output=self.layers(input)
        return output

def main():
    history_steps=50
    root_and_action=27
    adapt=AdaptationModule().to("cuda:0")
    #info_encoder=Info_Encoder().to("cuda:0")
    #ck_dict=torch.load("policy.pt")
    #info_encoder.load_state_dict(ck_dict["actor_encoder"])


    loss_func=nn.MSELoss()
    optimizer=torch.optim.Adam(adapt.parameters(),lr=0.0004)
    adapt.double()
    m=0
    for j in range(1):
        path_1="train/history"
        path_2="train/info"
        path_3="test/history"
        path_4="test/info"
        data_obs=pd.read_csv(path_1)
        data_obs=data_obs.values
        data_obs=torch.from_numpy(data_obs)
        data_ans=pd.read_csv(path_2)
        data_ans=data_ans.values
        data_ans=torch.from_numpy(data_ans)
        data_input=torch.cat([data_obs,data_ans],dim=-1)
                
        test_obs=pd.read_csv(path_3)
        test_obs=test_obs.values
        test_obs=torch.from_numpy(test_obs)
        test_ans=pd.read_csv(path_4)
        test_ans=test_ans.values
        test_ans=torch.from_numpy(test_ans)
        test_data_input=torch.cat([test_obs,test_ans],dim=-1)

        loss_sum=0

        for k in range(200):
            train_data=DataLoader(data_input,batch_size=64,shuffle=True)
            
            for i,data in enumerate(train_data):
                m=m+1
                data_fedin=data.to("cuda:0")
                train_obs,train_info=split(data_fedin,history_steps,root_and_action)
                train_input=train_obs.reshape(-1,history_steps,27)
                x=adapt(train_input)
                with torch.no_grad():
                    y=train_info
                loss=loss_func(x,y)
                loss_sum=loss_sum+loss
                if m % 100 == 0:
                    loss_average=loss_sum/100
                    print("train_loss ",m,":",loss_average.item())
                    loss_sum=0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if m==100000:
                    torch.save(adapt.state_dict(),"outputs/checkpoint_final_80.pt")
                    break
                if m % 1000 == 0:
                    with torch.no_grad(): 
                        test_data=DataLoader(test_data_input,batch_size=640,shuffle=True)
                        loss_sum_test=0
                        for l,test in enumerate(test_data):
                            if l==50:
                                break
                            test_fedin=test.to("cuda:0")
                            test_obs,test_info=split(test_fedin)
                            test_input=test_obs.reshape(-1,history_steps,root_and_action)
                            y_test=test_info
                            x_test=adapt(test_input)
                            test_loss=loss_func(x_test,y_test)
                            loss_sum_test=loss_sum_test+test_loss
                        print("eval ",m,":",loss_sum_test.item()/50)

if __name__=='__main__':
    main()



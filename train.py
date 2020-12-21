import torch.nn.functional as F 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import dataset as D
import loader as L 
import torchvision
import numpy as np
import nn as NN 
import torch 
import cv2
import os 


class AI:
    NAME    =   [
        "ando",
        "uemura",
        "enomaru",
        # "ooshima",
        "mizuki",
        "okamura",
        "kataoka",
        "kodama",
        "shinohara",
        "suetomo",
        "takemoto",
        # "tamejima",
        "nagao",
        "hamada",
        "masuda",
        # "matuzaki",
        "miyatake",
        "soushi",
        # "ryuuga",
        "yamaji",
        # "yamashita",
        "wada",
        "watanabe",
        # "teppei",
        "kawano",
        "higashi",
        # "tutiyama",
        # "toriyabe",
        # "matui",
        # "ishino",
    ]


    def __init__(
        self,
        EPOCH       =   40,
        IMAGE_SIZE  =   160,
        HIDDEN_1    =   320,
        LR          =   0.000005,
        model_num   =   20,
        TRAIN_DIR   =   "train-dataset",
        TEST_DIR    =   "test-dataset",
        PT_NAME     =   "nn.pt",
        LOSS_PNG    =   "loss.png",
        ACC_PNG     =   "acc.png"
        ):
        self.EPOCH      =   EPOCH
        self.IMAGE_SIZE =   IMAGE_SIZE
        self.HIDDEN_1   =   HIDDEN_1
        self.LR         =   LR 
        self.MODEL      =   NN.Net(num=20,inputSize=IMAGE_SIZE,Neuron=HIDDEN_1)
        self.OPTIMIZER  =   torch.optim.Adam(params=self.MODEL.parameters(),lr=self.LR)
        self.TRAIN_DIR  =   TRAIN_DIR
        self.TEST_DIR   =   TEST_DIR
        self.PT_NAME    =   PT_NAME
        self.LOSS_PNG   =   LOSS_PNG
        self.ACC_PNG    =   ACC_PNG

        ######------学習用データローダー-----#####
        train_data  =   L.Loader()
        train_data.setDir(self.TRAIN_DIR)
        self.train_data =   train_data.dataloader(self.IMAGE_SIZE)

        ######------推論用個別データローダー-----#####
        test_data   =   []
        for label,name in enumerate(self.NAME):
            # print(name)
            test_data.append(D.Dataset(
                test_dataset=self.TEST_DIR+"/"+name+"/",
                label=label,
                imgSize=self.IMAGE_SIZE
            ))

            test_data[label]    =   torch.utils.data.DataLoader(
                dataset=test_data[label],
                batch_size=1,
                shuffle=True
            )
        
        self.test_data  =   test_data

    def getTrainData(self):
        return self.train_data
    def getTestAndo(self):
        return self.test_data[0]
    def getTestUemura(self):
        return self.test_data[1]
    def getTestEnomaru(self):
        return self.test_data[2]
    def getTestOOshima(self):
        pass 
    def getTestMizuki(self):
        return self.test_data[3]
    def getTestOkamura(self):
        return self.test_data[4]
    def getTestKataoka(self):
        return self.test_data[5]
    def getTestKodama(self):
        return self.test_data[6]
    def getTestShinohara(self):
        return self.test_data[7]    
    def getTestSuetomo(self):
        return self.test_data[8]    
    def getTestTakemoto(self):
        return self.test_data[9]    
    def getTestTamejima(self):
        pass    
    def getTestNagao(self):
        return self.test_data[10]    
    def getTestHamada(self):
        return self.test_data[11]    
    def getTestMasuda(self):
        return self.test_data[12]    
    def getTestMatuzaki(self):
        pass    
    def getTestMiyatake(self):
        return self.test_data[13]    
    def getTestSoushi(self):
        return self.test_data[14]    
    def getTestRyuuga(self):
        pass 
    def getTestYamaji(self):
        return self.test_data[15]    
    def getTestYamashita(self):
        pass   
    def getTestWada(self):
        return self.test_data[16]    
    def getTestWatanabe(self):
        return self.test_data[17]    
    def getTestTeppei(self):
        pass
    def getTestKawano(self):
        return self.test_data[18]    
    def getTestHigashi(self):
        return self.test_data[19]    
    def getTestTutiyama(self):
        pass    
    def getTestToriyabe(self):
        pass 
    def getTestMatui(self):
        pass
    def getTestIshino(self):
        pass    
    
    def train(self,trainData):
        for data in tqdm(trainData):
            x,target    =   data 
            self.OPTIMIZER.zero_grad()
            output      =   self.MODEL(x)
            ######------損失関数-----#####
            loss = F.nll_loss(output,target)
            loss.backward()
            self.OPTIMIZER.step()
        return loss 
    
    def test(self,testData):
        #   学習停止
        self.MODEL.eval()
        total   =   0
        correct =   0
        cnt     =   0
        with torch.no_grad():
            for data in tqdm(testData):
                x,label =   data 
                m_x     =   torch.reshape(x[0][cnt],(1,1,160,160))
        
                output  =   self.MODEL(m_x.float())    
                _,p     =   torch.max(output.data,1)
                total   +=  label.size(0)
                correct +=  (p==label).sum().item()
                
                if cnt >= 200:
                    break
                else:
                    cnt +=  1
            print(100*correct/total)
            return 100*correct/total

    def test_old(self,testDir,label,imgSize,model):
        #   学習停止
        self.MODEL.eval()
        
        total   =   0
        correct =   0
        with torch.no_grad():
            for cnt,f in enumerate(os.listdir(testDir)):
                #   画像を読み込み
                img     =   cv2.imread(testDir+"/"+f)
                #   チャンネル数を1
                imgGray =   cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                #   リサイズ
                img     =   cv2.resize(imgGray,(imgSize,imgSize))
                # リシェイプ
                img = np.reshape(img,(1,imgSize,imgSize))
                # transpose h,c,w
                img = np.transpose(img,(1,2,0))
                # ToTensor 正規化される
                img = img.astype(np.uint8)
                mInput = transforms.ToTensor()(img) 
                mInput = mInput.view(-1, imgSize*imgSize)
                #   推論
                output = model(mInput)
                p = model.forward(mInput)
                if p.argmax() == label:
                    correct += 1
                total   +=  1
        return 100*correct/total

    def save_loss_png(self,loss):
        plt.figure()
        plt.plot(range(1,self.EPOCH+1),loss,label="trainLoss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(self.LOSS_PNG)

    def save_acc_png(self,acc,name):
        plt.figure()
        plt.plot(range(1,self.EPOCH+1),acc,label=label)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(str(name)+"_"+self.ACC_PNG)

    def save_model():
        torch.save(self.MODEL.state_dict(),self.PT_NAME)
        

def new_train():
    ai          =   AI(TRAIN_DIR="train",TEST_DIR="test")

    trainData       =   ai.getTrainData()
    test_ando       =   ai.getTestAndo()
    test_uemura     =   ai.getTestUemura()  
    test_enomaru    =   ai.getTestEnomaru()
    # test_ooshima    =   ai.getTestOOshima()
    test_mizuki     =   ai.getTestMizuki()
    test_okamura    =   ai.getTestOkamura()
    test_kataoka    =   ai.getTestKataoka()
    test_kodama     =   ai.getTestKodama()
    test_shinohara  =   ai.getTestShinohara()
    test_suetomo    =   ai.getTestSuetomo()
    test_takemoto   =   ai.getTestTakemoto()
    # test_tamejima   =   ai.getTestTamejima()
    test_nagao      =   ai.getTestNagao()
    test_hamada     =   ai.getTestHamada()
    test_masuda     =   ai.getTestMasuda()
    # test_matuzaki   =   ai.getTestMatuzaki()
    test_miyatake   =   ai.getTestMiyatake()
    test_soushi     =   ai.getTestSoushi()
    # test_ryuuga     =   ai.getTestRyuuga()
    test_yamaji     =   ai.getTestYamaji()
    # test_yamashita  =   ai.getTestYamashita()
    test_wada       =   ai.getTestWada()
    test_watanabe   =   ai.getTestWatanabe()
    # test_teppei     =   ai.getTestTeppei()
    test_kawano     =   ai.getTestKawano()
    test_higashi    =   ai.getTestHigashi()
    # test_tutiyama   =   ai.getTestTutiyama()
    # test_toriyabe   =   ai.getTestToriyabe()
    # test_matui      =   ai.getTestMatui()
    # test_ishino     =   ai.getTestIshino()


    loss    =   []
    ACC     =   {
        "ando":[],
        "uemura":[],
        "enomaru":[],
        "ooshima":[],
        "mizuki":[],
        "okamura":[],
        "kataoka":[],
        "kodama":[],
        "shinohara":[],
        "suetomo":[],
        "takemoto":[],
        "tamejima":[],
        "nagao":[],
        "hamada":[],
        "masuda":[],
        "matuzaki":[],
        "miyatake":[],
        "soushi":[],
        "ryuuga":[],
        "yamaji":[],
        "yamashita":[],
        "wada":[],
        "watanabe":[],
        "teppei":[],
        "kawano":[],
        "higashi":[],
        "tutiyama":[],
        "toriyabe":[],
        "matui":[],
        "ishino":[]
    }


    for e in range(ai.EPOCH):
        loss.append(ai.train(trainData))
        ando_acc        =   ai.test(test_ando)
        uemura_acc      =   ai.test(test_uemura)
        enomaru_acc     =   ai.test(test_enomaru)
        # ooshima_acc     =   ai.test(test_ooshima)
        mizuki_acc      =   ai.test(test_mizuki)
        okamura_acc     =   ai.test(test_okamura)
        kataoka_acc     =   ai.test(test_kataoka)
        kodama_acc      =   ai.test(test_kodama)
        shinohara_acc   =   ai.test(test_shinohara)
        suetomo_acc     =   ai.test(test_suetomo)
        takemoto_acc    =   ai.test(test_takemoto)
        # tamejima_acc    =   ai.test(test_tamejima)
        nagao_acc       =   ai.test(test_nagao)
        hamada_acc      =   ai.test(test_hamada)
        masuda_acc      =   ai.test(test_masuda)
        # matuzaki_acc    =   ai.test(test_matuzaki)
        miyatake_acc    =   ai.test(test_miyatake)
        soushi_acc      =   ai.test(test_soushi)
        # ryuuga_acc      =   ai.test(test_ryuuga)
        yamaji_acc      =   ai.test(test_yamaji)
        # yamashita_acc   =   ai.test(test_yamashita)
        wada_acc        =   ai.test(test_wada)
        watanabe_acc    =   ai.test(test_watanabe)
        # teppei_acc      =   ai.test(test_teppei)
        kawano_acc      =   ai.test(test_kawano)
        higashi_acc     =   ai.test(test_higashi)
        # tutiyama_acc    =   ai.test(test_tutiyama)
        # toriyabe_acc    =   ai.test(test_toriyabe)
        # matui_acc       =   ai.test(test_matui)
        # ishino_acc      =   ai.test(test_ishino)
        
        
        ACC["ando"].append(ando_acc)
        ACC["uemura"].append(uemura_acc)
        ACC["enomaru"].append(enomaru_acc)
        # ACC["ooshima"].append(ooshima_acc)
        ACC["mizuki"].append(mizuki_acc)
        ACC["okamura"].append(okamura_acc)
        ACC["kataoka"].append(kataoka_acc)
        ACC["kodama"].append(kodama_acc)
        ACC["shinohara"].append(shinohara_acc)
        ACC["suetomo"].append(suetomo_acc)
        ACC["takemoto"].append(takemoto_acc)
        # ACC["tamejima"].append(tamejima_acc)
        ACC["nagao"].append(nagao_acc)
        ACC["hamada"].append(hamada_acc)
        ACC["masuda"].append(masuda_acc)
        # ACC["matuzaki"].append(matuzaki_acc)
        ACC["miyatake"].append(miyatake_acc)
        ACC["soushi"].append(soushi_acc)
        # ACC["ryuuga"].append(ryuuga_acc)
        ACC["yamaji"].append(yamaji_acc)
        # ACC["yamashita"].append(yamashita_acc)
        ACC["wada"].append(wada_acc)
        ACC["watanabe"].append(watanabe_acc)
        # ACC["teppei"].append(teppei_acc)
        ACC["kawano"].append(kawano_acc)
        ACC["higashi"].append(higashi_acc)
        # ACC["tutiyama"].append(acc)
        # ACC["toriyabe"].append(acc)
        # ACC["matui"].append(acc)
        # ACC["ishino"].append(acc)

    ai.save_loss_png(loss)
    ai.save_acc_png(ACC["ando"],"ando")
    ai.save_acc_png(ACC["uemura"],"uemura")
    ai.save_acc_png(ACC["enomaru"],"enomaru")
    # ai.save_acc_png(ACC["ooshima"],"ooshima")
    ai.save_acc_png(ACC["mizuki"],"mizuki")
    ai.save_acc_png(ACC["okamura"],"okamura")
    ai.save_acc_png(ACC["kataoka"],"kataoka")
    ai.save_acc_png(ACC["kodama"],"kodama")
    ai.save_acc_png(ACC["shinohara"],"shinohara")
    ai.save_acc_png(ACC["suetomo"],"suetomo")
    ai.save_acc_png(ACC["takemoto"],"takemoto")
    # ai.save_acc_png(ACC["tamejima"],"tamejima")
    ai.save_acc_png(ACC["nagao"],"nagao")
    ai.save_acc_png(ACC["hamada"],"hamada")
    ai.save_acc_png(ACC["masuda"],"masuda")
    # ai.save_acc_png(ACC["matuzaki"],"matuzaki")
    ai.save_acc_png(ACC["miyatake"],"miyatake")
    ai.save_acc_png(ACC["soushi"],"soushi")
    # ai.save_acc_png(ACC["ryuuga"],"ryuuga")
    ai.save_acc_png(ACC["yamaji"],"yamaji")
    # ai.save_acc_png(ACC["yamashita"],"yamashita")
    ai.save_acc_png(ACC["wada"],"wada")
    ai.save_acc_png(ACC["watanabe"],"watanabe")
    # ai.save_acc_png(ACC["teppei"],"teppei")
    ai.save_acc_png(ACC["kawano"],"kawano")
    ai.save_acc_png(ACC["higashi"],"higashi")
    # ai.save_acc_png(ACC["tutiyama"],"tutiyama")
    # ai.save_acc_png(ACC["toriyabe"],"toriyabe")
    # ai.save_acc_png(ACC["matui"],"matui")
    # ai.save_acc_png(ACC["ishino"],"ishino")

    ai.save_model()

def old_train():
    test_root_dir       =   "test"
    test_ando_dir       =   test_root_dir+"/"+"ando/"
    test_uemura_dir     =   test_root_dir+"/"+"uemura/"
    test_enomaru_dir    =   test_root_dir+"/"+"enomaru/"
    # test_ooshima_dir    =   test_root_dir+"/"+"ooshima/"
    test_mizuki_dir     =   test_root_dir+"/"+"mizuki/"
    test_okamura_dir    =   test_root_dir+"/"+"okamura/"
    test_kataoka_dir    =   test_root_dir+"/"+"kataoka/"
    test_kodama_dir     =   test_root_dir+"/"+"kodama/"
    test_shinohara_dir  =   test_root_dir+"/"+"shinohara/"
    test_suetomo_dir    =   test_root_dir+"/"+"suetomo/"
    test_takemoto_dir   =   test_root_dir+"/"+"takemoto/"
    # test_tamejima_dir   =   test_root_dir+"/"+"tamejima/"
    test_nagao_dir      =   test_root_dir+"/"+"nagao/"
    test_hamada_dir     =   test_root_dir+"/"+"hamada/"
    test_masuda_dir     =   test_root_dir+"/"+"masuda/"
    # test_matuzaki_dir   =   test_root_dir+"/"+"matuzaki/"
    test_miyatake_dir   =   test_root_dir+"/"+"miyatake/"
    test_soushi_dir     =   test_root_dir+"/"+"soushi/"
    # test_ryuuga_dir     =   test_root_dir+"/"+"ryuuga/"
    test_yamaji_dir     =   test_root_dir+"/"+"yamaji/"
    # test_yamashita_dir  =   test_root_dir+"/"+"yamashita/"
    test_wata_dir       =   test_root_dir+"/"+"wada/"
    test_watanabe_dir   =   test_root_dir+"/"+"watanabe/"
    # test_teppei_dir     =   test_root_dir+"/"+"teppei/"
    test_kawano_dir     =   test_root_dir+"/"+"kawano/"
    test_higashi_dir    =   test_root_dir+"/"+"higashi/"
    # test_tutiyama_dir   =   test_root_dir+"/"+"tutiyama/"
    # test_toriyabe_dir   =   test_root_dir+"/"+"toriyabe/"
    # test_matui_dir      =   test_root_dir+"/"+"matui/"
    # test_ishino_dir     =   test_root_dir+"/"+"ishino/"


    ai          =   AI(TRAIN_DIR="train",TEST_DIR=test_root_dir)

    train_data  =   ai.getTrainData()

    loss        =   []
    for e in range(ai.EPOCH):
        loss.append(ai.train(train_data))
        ai.test_old(test_ando_dir,0,ai.IMAGE_SIZE,ai.MODEL)
        ai.test_old(test_uemura_dir,1,ai.IMAGE_SIZE,ai.MODEL)
        ai.test_old(test_enomaru_dir,2,ai.IMAGE_SIZE,ai.MODEL)
        


if __name__ == "__main__":
    # new_train()
    old_train()




















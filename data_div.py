import shutil
import os 

class DataSplit:
    def __init__(self,read_dir,save_dir,train_data_max=1000):
        dataMax,nameList    =   self.__read(read_dir)
        self.__save(save_dir,dataMax,nameList,train_data_max)
        pass 

    def __del__(self):
        pass 

    def __read(self,read_dir):
        try:
            dataMax     =   len(os.listdir(read_dir))
            fileNameList=   []
            for f in os.listdir(read_dir):
                fileNameList.append(read_dir+f)
            return dataMax,fileNameList
        except FileNotFoundError:
            return 0,""

    def __save(self,save_dir,dataMax,nameList,trainDataMax):
        train_dir   =   "train/"    +save_dir
        test_dir    =   "test/"     +save_dir
        try:
            os.makedirs(train_dir)
            os.makedirs(test_dir)
        except FileExistsError:
            #   もし学習用データ数と推論用データ数の合計が、
            #   読み込んだファイルの数よりも多いなら、
            #   上書きを禁止する。
            #   同じときまたは少ないときは上書き
            if len(os.listdir(train_dir)) + len(os.listdir(test_dir)) > dataMax:
                return 0
        
        #   もし読み込むデータ総数が、学習用に保存するデータ数未満だったら
        if dataMax  <   trainDataMax:
            print("{:<30s}に保存するデータが足りません".format(train_dir))
            print("保存したいデータ数：{:>8d}".format(int(trainDataMax)))
            print("読み込んだデータ数：{:>8d}".format(int(dataMax)))
        else:
            cnt =   0
            for path in nameList:
                if cnt < trainDataMax:
                    # print(train_dir+str(cnt)+".jpg")
                    savePath    =   train_dir+str(cnt)+".jpg"
                else:
                    savePath    =   test_dir+str(cnt)+".jpg"
                
                #   保存
                shutil.copyfile(path,savePath)

                cnt +=  1
        pass 

class RunClasses30:
    NAME    =   [
        "ando",
        "uemura",
        "enomaru",
        "ooshima",
        "mizuki",
        "okamura",
        "kataoka",
        "kodama",
        "shinohara",
        "suetomo",
        "takemoto",
        "tamejima",
        "nagao",
        "hamada",
        "masuda",
        "matuzaki",
        "miyatake",
        "soushi",
        "ryuuga",
        "yamaji",
        "yamashita",
        "wada",
        "watanabe",
        "teppei",
        "kawano",
        "higashi",
        "tutiyama",
        "toriyabe",
        "matui",
        "ishino",
    ]
    def __init__(self,read_root_dir,save_root_dir,train_data_max=1000):
        for name in self.NAME:
            ds  =   DataSplit(
                read_dir        =read_root_dir  +name   +"/",
                save_dir        =save_root_dir  +name   +"/",
                train_data_max  =train_data_max,
            )
        pass 

    def __del__(self):
        pass 


if __name__ == "__main__":
    rc  =   RunClasses30(
        read_root_dir   =   "dataset/",
        save_root_dir   =   "",
        train_data_max  =   900
    )
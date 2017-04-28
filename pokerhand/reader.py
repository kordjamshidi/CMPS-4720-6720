import csv

# files for input and output
trainfile="train.csv"
testfile="test1.csv"

#reads the file and returns the data in raw form
def filereader(file, tt):
    with open(file,'rb') as f:
        reader = csv.reader(f)
        data_list = list(reader)
        #might need to pop the header
        data_list.pop(0)
        for i in range(len(data_list)):
            if(tt):
                data_list[i].pop(0)
                
            for j in range(len(data_list[i])):
                data_list[i][j]=int(data_list[i][j])
    return data_list
                    
train_data=filereader(trainfile,False)
test_input=filereader(testfile,True)

from __future__ import division
import numpy as np

import reader
import train

TEST_SIZE = 200002

# load train results and answer
layer_3 = train.layer_3
testset = reader.test_input
ans = reader.ans

# cross validation
def crossvalidate(s):
    sum = 0
    for i in range(5):
        weights = None
        test_inputs = testset[i][:, :5]
        test_inputs = test_inputs.astype(np.float32)
        test_answer = testset[i][:, 5]
        test_answer = code_answer(test_answer)

        for j in range(5):
            if i == j:
                continue
            train_inputs = testset[j][:, :5]
            train_inputs = train_inputs.astype(np.float32)
            train_answer = testset[j][:, 5]
            train_answer = code_answer(train_answer)
            weights = train(train_inputs, train_answer, structure, weights)
        
        error = test(test_inputs, test_answer, weights)
        f.write("Number of misclassifications in fold " + str(i) + ": " + str(error) + "\n")
        sum += error
        print weights

    f.write("Average: " + str(sum/5.0) + "\n")
    print "Average: " + str(sum/5.0) + "\n"


# initialize stats vars
cls = [0 for x in range(TEST_SIZE)]
error = 0
positive_0 = 0
positive_1 = 0
positive_2 = 0
positive_3 = 0
positive_4 = 0
positive_5 = 0
positive_6 = 0
positive_7 = 0
positive_8 = 0
positive_9 = 0
negative_0 = 0
negative_1 = 0
negative_2 = 0
negative_3 = 0
negative_4 = 0
negative_5 = 0
negative_6 = 0
negative_7 = 0
negative_8 = 0
negative_9 = 0
fp_0 = 0
fp_1 = 0
fp_2 = 0
fp_3 = 0
fp_4 = 0
fp_5 = 0
fp_6 = 0
fp_7 = 0
fp_8 = 0
fp_9 = 0
fn_0 = 0
fn_1 = 0
fn_2 = 0
fn_3 = 0
fn_4 = 0
fn_5 = 0
fn_6 = 0
fn_7 = 0
fn_8 = 0
fn_9 = 0

for i in range(len(layer_3)):
    for j in range(10):
        if int(layer_3[i][j]) == 1:
            cls[i] = j
            # positive
            if j==0:
                positive_0+=1
            elif j==1:
                positive_1+=1
            elif j==2:
                positive_2+=1
            elif j==3:
                positive_3+=1
            elif j==4:
                positive_4+=1
            elif j==5:
                positive_5+=1
            elif j==6:
                positive_6+=1
            elif j==7:
                positive_7+=1
            elif j==8:
                positive_8+=1
            else:
                positive_9+=1
        else:
            # negative
            if j==0:
                negative_0+=1
            elif j==1:
                negative_1+=1
            elif j==2:
                negative_2+=1
            elif j==3:
                negative_3+=1
            elif j==4:
                negative_4+=1
            elif j==5:
                negative_5+=1
            elif j==6:
                negative_6+=1
            elif j==7:
                negative_7+=1
            elif j==8:
                negative_8+=1
            else:
                negative_9+=1
                
            
    if cls[i]!=int(ans[i][0]): #false
        error+=1
        # false positive
        if cls[i]==0:
            fp_0+=1
        elif cls[i]==1:
            fp_1+=1
        elif cls[i]==2:
            fp_2+=1
        elif cls[i]==3:
            fp_3+=1
        elif cls[i]==4:
            fp_4+=1
        elif cls[i]==5:
            fp_5+=1
        elif cls[i]==6:
            fp_6+=1
        elif cls[i]==7:
            fp_7+=1
        elif cls[i]==8:
            fp_8+=1
        else:
            fp_9+=1
            
        # false negative
        if cls[i]!=0:
            fn_0+=1
        elif cls[i]!=1:
            fn_1+=1
        elif cls[i]!=2:
            fn_2+=1
        elif cls[i]!=3:
            fn_3+=1
        elif cls[i]!=4:
            fn_4+=1
        elif cls[i]!=5:
            fn_5+=1
        elif cls[i]!=6:
            fn_6+=1
        elif cls[i]!=7:
            fn_7+=1
        elif cls[i]!=8:
            fn_8+=1
        else:
            fn_9+=1

# true positive
tp_0 = positive_0-fp_0
tp_1 = positive_1-fp_1
tp_2 = positive_2-fp_2
tp_3 = positive_3-fp_3
tp_4 = positive_4-fp_4
tp_5 = positive_5-fp_5
tp_6 = positive_6-fp_6
tp_7 = positive_7-fp_7
tp_8 = positive_8-fp_8
tp_9 = positive_9-fp_9
# true negative
tn_0 = negative_0-fn_0
tn_1 = negative_1-fn_1
tn_2 = negative_2-fn_2
tn_3 = negative_3-fn_3
tn_4 = negative_4-fn_4
tn_5 = negative_5-fn_5
tn_6 = negative_6-fn_6
tn_7 = negative_7-fn_7
tn_8 = negative_8-fn_8
tn_9 = negative_9-fn_9
        
print("error", error)
print("Accuracy", 1-error/TEST_SIZE)
print("TP",(tp_0+tp_1+tp_2+tp_3+tp_4+tp_5+tp_6+tp_7+tp_8+tp_9)/10)
print("TN",(tn_0+tn_1+tn_2+tn_3+tn_4+tn_5+tn_6+tn_7+tn_8+tn_9)/10)
print("FP",(fp_0+fp_1+fp_2+fp_3+fp_4+fp_5+fp_6+fp_7+fp_8+fp_9)/10)
print("FN",(fn_0+fn_1+fn_2+fn_3+fn_4+fn_5+fn_6+fn_7+fn_8+fn_9)/10)
precision_0 = tp_0/(tp_0+fp_0)
precision_1 = tp_1/(tp_1+fp_1)
precision_2 = tp_2/(tp_2+fp_2)
precision_3 = tp_3/(tp_3+fp_3)
precision_4 = tp_4/(tp_4+fp_4)
precision_5 = tp_5/(tp_5+fp_5)
precision_6 = tp_6/(tp_6+fp_6)
precision_7 = tp_7/(tp_7+fp_7)
precision_8 = tp_8/(tp_8+fp_8)
precision_9 = tp_9/(tp_9+fp_9)
precision = (precision_0+precision_1+precision_2+precision_3+precision_4+precision_5+precision_6+precision_7+precision_8+precision_9)/10

recall_0 = tp_0/(tp_0+fn_0)
recall_1 = tp_0/(tp_1+fn_1)
#recall_2 = tp_0/(tp_2+fn_2)
#recall_3 = tp_0/(tp_3+fn_3)
#recall_4 = tp_0/(tp_4+fn_4)
#recall_5 = tp_0/(tp_5+fn_5)
#recall_6 = tp_0/(tp_6+fn_6)
#recall_7 = tp_0/(tp_7+fn_7)
#recall_8 = tp_0/(tp_8+fn_8)
#recall_9 = tp_0/(tp_9+fn_9)
recall = (recall_0+recall_1)/2

print("precision", precision)
print("recall", recall)

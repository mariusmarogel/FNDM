import numpy as np
for model in ['RNN', 'BiRNN', 'GRU', 'BiGRU', 'LSTM', 'BiLSTM']:
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    for i in range(1, 11):
        # Change filename path for every set of results
        filename = "output" + str(i) + ".txt"
        with open(filename, 'r') as f:
            c = f.read()
            ## accuracy
            s1 = c.split('Model ' + model  + ' Test Accuracy: ')[1].split('accuracy                           ')[1].split(' ')[0]
            ## precision
            s2 = c.split('Model ' + model  + ' Test Accuracy: ')[1].split('macro avg       ')[1].split('      ')[0]
            ##recall
            s3 = c.split('Model ' + model  + ' Test Accuracy: ')[1].split('macro avg       ')[1].split('      ')[1]
            ##f1
            s4 = c.split('Model ' + model  + ' Test Accuracy: ')[1].split('macro avg       ')[1].split('      ')[2]
            arr1.append(float(s1))
            arr2.append(float(s2))
            arr3.append(float(s3))
            arr4.append(float(s4))
    v1 = "{:.3f}".format(np.mean(np.array(arr1)))
    v2 = "{:.3f}".format(np.mean(np.array(arr2)))
    v3 = "{:.3f}".format(np.mean(np.array(arr3)))
    v4 = "{:.3f}".format(np.mean(np.array(arr4)))
    st1 = "{:.3f}".format(np.std(arr1))
    st2 = "{:.3f}".format(np.std(arr2))
    st3 = "{:.3f}".format(np.std(arr3))
    st4 = "{:.3f}".format(np.std(arr4))
    with open('out.txt', 'a') as f:
        f.write(str(v1) + "+/-" + str(st1) + "," + str(v2) + "+/-" + str(st2) + "," + str(v3) + "+/-" + str(st3) + "," + str(v4) + "+/-" + str(st4))
        f.write("\n")




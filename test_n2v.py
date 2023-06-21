import numpy as np

files = ['11.txt', '105.txt', '051.txt', '0505.txt', '21.txt', '12.txt']
for filename in files:
    with open('100d/' + filename, "r") as f:
        arr1 = []
        arr2 = []
        arr3 = []
        arr4 = []
        c = f.read()
        i = 1
        s = c.split('\n')
        for line in s:
            if 'accuracy' in line:
                a = line.split('    accuracy                           ')[1].split(' ')[0]
                arr1.append(float(a))
            if 'macro avg' in line:
                b = line.split('   macro avg       ')[1].split(' ')[0]
                arr2.append(float(b))
                c = line.split('   macro avg       ')[1].split('      ')[1]
                arr3.append(float(c))
                d = line.split('   macro avg       ')[1].split('      ')[2]
                arr4.append(float(d))
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
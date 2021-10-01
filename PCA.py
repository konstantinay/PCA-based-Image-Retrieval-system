import cv2
import os
import numpy as np

#pairnoume tis listes me ta onomata apo tous fakelous
train_image_names = os.listdir('./DataBase')
train_image_names.sort()
test_image_names = os.listdir('./test')
test_image_names.sort()

train_images = []
test_images = []

#prosthetoume se kathe lista ta dianusmata eikonwn
for img_name in train_image_names:
    if img_name=='Thumbs.db':
        continue
    train_images.append(cv2.imread('./DataBase/'+img_name,cv2.IMREAD_GRAYSCALE).flatten()) 
    
for img_name in test_image_names:
    test_images.append(cv2.imread('./test/'+img_name,cv2.IMREAD_GRAYSCALE).flatten()) 
    
train_images = np.array(train_images, dtype=np.double)
test_images = np.array(test_images, dtype=np.double)

#kanonikopoioume tis times twn eikonwn sto diasthma [0,1]
train_images /= 255
test_images /= 255

#ftiaxnoume thn mesh eikona & thn afairoume apo ta dedomena
mean_image = np.expand_dims(train_images.mean(0),0)

train_images -= mean_image
test_images -= mean_image

#kanoume tis grammes sthles
train_images = train_images.transpose(1,0)
test_images = test_images.transpose(1,0)

##### PCA #####
A = train_images

#ypologizoume to Atranspose*A
ATA = A.transpose(1,0)@A

#kanoume SVD kai kratame ta idiodianysmata tou Atranspose*A
V, _, _ = np.linalg.svd(ATA)

#provaloume ta dedomena sta idiodianysmata tou Atranspose*A wste na paroume ta 
#idiodianysmata toy A*Atranspose
U = A@V

#kanonikopoioume wste kathe sthlh na exei monadiaia norma l1
U /= U.mean(0)


############ k = 100 ############

#kratame k=100 sthles tou U
U_100 = U

#vriskoume thn anaparastash kathe train eikonas
W_train_100 = U_100.transpose(1,0)@A

#vriskoume thn anaparastash kathe test eikonas
W_test_100 = U_100.transpose(1,0)@test_images

success = 0
#pairnoume ka8e ena apo ta ta test dedomena 
for i in range(W_test_100.shape[1]):
    query = W_test_100[:,i]
    diffs = np.zeros((100))

    #ypologizoume to mse tou erwthmatos me ola ta dedomena ths vashs
    for j in range(W_train_100.shape[1]):
        diffs[j] = ((query - W_train_100[:,j])**2).mean()

    #vriskoume thn elaxisth apostash
    min_err = diffs.min()

    diffs = list(diffs)

    #elegxoume an h eikona me thn elaxisth apostash apo to erwthma einai
    #h swsth kai ananewnoume to success
    if train_image_names[diffs.index(min_err)] == test_image_names[i]:
        success += 1
    
print('Success rate for k=100:',success/11*100,"%")


############ k = 50 ############

#kratame k=50 sthles tou U
U_50 = U
U_50[:,50:] = 0

#vriskoume thn anaparastash kathe train eikonas
W_train_50 = U_50.transpose(1,0)@A

#vriskoume thn anaparastash kathe test eikonas
W_test_50 = U_50.transpose(1,0)@test_images

success = 0
#pairnoume ta test dedomena 
for i in range(W_test_50.shape[1]):
    query = W_test_50[:,i]
    diffs = np.zeros((100))

    #ypologizoume to mse tou erwthmatos me ola ta dedomena ths vashs
    for j in range(W_train_50.shape[1]):
        diffs[j] = ((query - W_train_50[:,j])**2).mean()

    #vriskoume thn elaxisth apostash
    min_err = diffs.min()

    diffs = list(diffs)

    #elegxoume an h eikona me thn elaxisth apostash apo to erwthma einai
    #h swsth kai ananewnoume to success
    if train_image_names[diffs.index(min_err)] == test_image_names[i]:
        success += 1
    
print('Success rate for k=50:',success/11*100,'%')


######## k = 10 ########

#kratame k=10 sthles tou U
U_10 = U
U_10[:,10:] = 0

#vriskoume thn anaparastash kathe train eikonas
W_train_10 = U_10.transpose(1,0)@A

#vriskoume thn anaparastash kathe test eikonas
W_test_10 = U_10.transpose(1,0)@test_images

success = 0
#pairnoume ta test dedomena 
for i in range(W_test_10.shape[1]):
    query = W_test_10[:,i]

    diffs = np.zeros((100))

    #ypologizoume to mse tou erwthmatos me ola ta dedomena ths vashs
    for j in range(W_train_10.shape[1]):
        diffs[j] = ((query - W_train_10[:,j])**2).mean()

    #vriskoume thn elaxisth apostash
    min_err = diffs.min()

    diffs = list(diffs)

    #elegxoume an h eikona me thn elaxisth apostash apo to erwthma einai
    #h swsth kai ananewnoume to success
    if train_image_names[diffs.index(min_err)] == test_image_names[i]:
        success += 1
    
print('Success rate for k=10:',success/11*100,'%')
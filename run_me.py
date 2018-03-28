# Import modules

import numpy as np
from scipy import misc
from scipy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from PIL import Image
from sklearn.cluster import KMeans

def read_scene():
    data_x = misc.imread('../../Data/Scene/times_square.jpg')

    return (data_x)

def read_faces():
    nFaces = 100
    nDims = 2500
    data_x = np.empty((0, nDims), dtype=float)

    for i in np.arange(nFaces):
        data_x = np.vstack((data_x, np.reshape(misc.imread('../../Data/Faces/face_%s.png' % (i)), (1, nDims))))

    return (data_x)

if __name__ == '__main__':

    ################################################
    # PCA

    data_x = read_faces()
    print('X = ', data_x.shape)

    img_original = Image.fromarray(data_x[1,:].reshape(50,50))

    print('Implement PCA here ...')

    N = 100
    p = 2500

    C_x = np.zeros((2500,2500))

    for i in range(100):
        C_x = np.add(C_x,data_x.T.dot(data_x))

    C_x = C_x/100

    print("C_x shape = ",C_x.shape)

    k_list = [3,5,10,30,50,100,150,300]
    ax_array = [i for i in range(1,9)]

    for index,k in enumerate(k_list):
        v,w = LA.eigh(C_x,eigvals = (C_x.shape[0]-k,C_x.shape[0]-1))

        print("Eigen Value V shape = ",v.shape)
        print("Eigen Vector W = ", w.shape)

        # if (w.dot(w.T) - np.eye(2500)).all():
        #     print("Identity Matrix ")
        # else:
        #     print("Identity Matrix ")

        X_projected = data_x.dot(w)

        X_recon = data_x.dot(w).dot(w.T)

        print("Reconstruction Image shape = ", X_recon.shape)

        print("X_recon[1,:] shape = ",X_recon[1,:].shape)

        X_for_image = X_recon[1,:].reshape(50,50)

        print("X_for_image shape = ", X_for_image.shape)

        print("\nReconstruction error in PCA for k = ",k," = ",np.sqrt(1/(N*p))*LA.norm(data_x - X_recon))

        compression_rate_PCA = (X_projected.nbytes + w.nbytes)/data_x.nbytes
        print("\nCompression_rate PCA for k = ", k, " = ", compression_rate_PCA)

        img = Image.fromarray(X_for_image)

        img.show()



    plt.figure(figsize=(9, 9))

    plt.subplot(331)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/PCA Original.png'))
    plt.xlabel('PCA Original')

    plt.subplot(332)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/PCA K = 3.png'))
    plt.xlabel('PCA K = 3')

    plt.subplot(333)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/PCA K = 5.png'))
    plt.xlabel('PCA K = 5')

    plt.subplot(334)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/PCA K = 10.png'))
    plt.xlabel('PCA K = 10')

    plt.subplot(335)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/PCA K = 30.png'))
    plt.xlabel('PCA K = 30')

    plt.subplot(336)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/PCA K = 50.png'))
    plt.xlabel('PCA K = 50')

    plt.subplot(337)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/PCA K = 100.png'))
    plt.xlabel('PCA K = 100')

    plt.subplot(338)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/PCA K = 150.png'))
    plt.xlabel('PCA K = 150')

    plt.subplot(339)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/PCA K = 300.png'))
    plt.xlabel('PCA K = 300')

    plt.show()


    ################################################
    # K-Means

    K_Mean_Vals = [2, 5, 10, 25, 50, 75, 100, 200]

    data_x = read_scene()
    print('X = ', data_x.shape)

    flattened_image = data_x.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
    print('Flattened image = ', flattened_image.shape)
    #
    plt.imshow(flattened_image.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2]))
    plt.xlabel('KMeans Original times_square')
    plt.xticks(())
    plt.yticks(())
    plt.show()
    #
    print('Implement k-means here ...')


    SSE_arr = np.empty(0)

    for i,k in enumerate(K_Mean_Vals):
        kmeans = KMeans(n_clusters=k).fit(flattened_image)

        print("K-means labels ",kmeans.labels_)


        centroid_array = kmeans.cluster_centers_
        print(" Cluster Center shape = ",centroid_array.shape)
        print(" Cluster Center array = ",centroid_array)

        labels = kmeans.predict(flattened_image)
        print(" Labels shape = ", labels.shape)

        reconstructed_image = np.copy(flattened_image)
        for i,label in enumerate(labels):
            reconstructed_image[i] = centroid_array[label]


        reconstructed_image = reconstructed_image.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
        print('Reconstructed image = ', reconstructed_image.shape)

        Reconstruction_error_kmeans = np.sqrt(np.mean(np.square(np.ravel(flattened_image) - np.ravel(reconstructed_image))))
        print("\nReconstruction error KMeans for k = ", k, " = ",Reconstruction_error_kmeans)

        compression_rate_kmeans = (k*32*3 + np.ceil(np.log2(k))*400*400)/(24*400*400)
        print("\nCompression_rate for k in K-Means = ", k, " = ",compression_rate_kmeans)

        SSE = np.sum(np.square(flattened_image.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2]) - reconstructed_image))
        SSE_arr = np.append(SSE_arr,SSE)

        plt.imshow(reconstructed_image)
        string = "K = " + str(k)
        plt.xlabel(string)
        plt.xticks(())
        plt.yticks(())
        plt.show()

    print("SSE_array = ",SSE_arr)
    plt.plot(K_Mean_Vals,SSE_arr, 'or-', linewidth=3)
    plt.ylabel("SSE")  # Y-axis label
    plt.xlabel("K for K-Means")  # X-axis label
    plt.title("Elbow Plot")  # Plot title

    plt.show()


    plt.figure(figsize=(9, 9))

    plt.subplot(331)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/KMeans Original times_square.png'))
    plt.xlabel('KMeans Original times_square')

    plt.subplot(332)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/KMeans K = 2.png'))
    plt.xlabel('KMeans K = 2')

    plt.subplot(333)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/KMeans K = 5.png'))
    plt.xlabel('KMeans K = 5')

    plt.subplot(334)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/KMeans K = 10.png'))
    plt.xlabel('KMeans K = 10')

    plt.subplot(335)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/KMeans K = 25.png'))
    plt.xlabel('KMeans K = 25')


    plt.subplot(336)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/KMeans K = 50.png'))
    plt.xlabel('KMeans K = 50')


    plt.subplot(337)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/KMeans K = 75.png'))
    plt.xlabel('KMeans K = 75')


    plt.subplot(338)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/KMeans K = 100.png'))
    plt.xlabel('KMeans K = 100')

    plt.subplot(339)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(Image.open('../../Submission/Figures/KMeans K = 200.png'))
    plt.xlabel('KMeans K = 200')

    plt.show()


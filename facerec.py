#module for generating eigenfaces map for AT&T Face Database
#a dataset of image is read and a pdf output of the eigen-map is generated.


# -*- coding: utf-8 -*-
import os
import numpy as np
# import tinyfacerec modules
import errno


import sys
import Image as Image
def read_images ( path , sz = None ) :
    c = 0
    X , y = [] , []
    for dirname , dirnames , filenames in os.walk ( path ) :
        #print dirname,dirnames,filenames
        for subdirname in dirnames :
            subject_path = os.path.join ( dirname , subdirname )
            for filename in os.listdir ( subject_path ) :
                #print filename
                try :
                    im = Image.open ( os.path.join ( subject_path , filename ) )
                    im = im.convert ( "L" )
                    #print im
                    # resize to given size ( if given )
                    if ( sz is not None ) :
                        im = im.resize ( sz , Image.ANTIALIAS )
                    X.append ( np.asarray ( im , dtype = np.uint8 ) )
                    y.append ( c )
                        
                except IOError :
                    print "IO error"
                    #print " I / O error ({0}) : {1} ".format ( errno , strerror )
                except :
                    print " Unexpected error : " , sys.exc_info () [0]
                    raise
            c = c +1
    return [X , y ]



def asRowMatrix ( X ) :
    if len ( X ) == 0:
        return np.array ([])
    mat = np.empty ((0 , X [0]. size ) , dtype = X [0]. dtype )
    for row in X :
        mat = np.vstack (( mat , np.asarray ( row ).reshape (1 , -1) ) )
    return mat


def asColumnMatrix ( X ) :
    if len ( X ) == 0:
        return np.array ([])
    mat = np.empty (( X [0]. size , 0) , dtype = X [0]. dtype )
    for col in X :
        mat = np.hstack (( mat , np.asarray ( col ).reshape ( -1 ,1) ) )
    return mat


def pca (X , y , num_components =0) :
    #print X
    [n,d] = X.shape
    if ( num_components <= 0) or ( num_components > n ) :
        num_components = n
    mu = X.mean( axis =0)
    X = X - mu
    if n > d :
        C = np.dot ( X .T , X )
        [ eigenvalues , eigenvectors ] = np.linalg.eigh ( C )
    else :
        C = np.dot (X , X.T )
        [ eigenvalues , eigenvectors ] = np.linalg.eigh ( C )
        eigenvectors = np.dot ( X .T , eigenvectors )
        for i in xrange ( n ) :
            eigenvectors [: , i ] = eigenvectors [: , i ]/ np.linalg.norm ( eigenvectors [: , i ])
# or simply perform an economy size decomposition
# eigenvectors , eigenvalues , variance = np.linalg.svd ( X .T , full_matrices = False )
# sort eigenvectors descending by their eigenvalue
    idx = np.argsort ( - eigenvalues )
    eigenvalues = eigenvalues [ idx ]
    eigenvectors = eigenvectors [: , idx ]
# select only num_c omponen ts
    eigenvalues = eigenvalues [0: num_components ]. copy ()
    eigenvectors = eigenvectors [: ,0: num_components ]. copy ()
    return [ eigenvalues , eigenvectors , mu ]

def normalize (X , low , high , dtype = None ) :
    X = np.asarray ( X )
    minX , maxX = np.min ( X ) , np.max ( X )
    # normalize to [0...1].
    X = X - float ( minX )
    X = X / float (( maxX - minX ) )
# scale to [ low ... high ].
    X = X * ( high - low )
    X = X + low
    if dtype is None :
        return np.asarray ( X )
    return np.asarray (X , dtype = dtype )


def project (W , X , mu = None ) :
    if mu is None :
        return np.dot (X , W )
    return np.dot ( X - mu , W )

def reconstruct (W , Y , mu = None ) :
    if mu is None :
        return np.dot (Y , W.T )
    return np.dot (Y , W.T ) + mu


# append tinyfacerec to module search path


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def create_font ( fontname = 'Tahoma' , fontsize =10) :
    return { 'fontname': fontname , 'fontsize': fontsize }

def subplot ( title , images , rows , cols , sptitle = " subplot " , sptitles =[] , colormap = cm .
    gray , ticks_visible = True , filename = None ) :
    fig = plt.figure ()
# main title
    fig.text (.5 , .95 , title , horizontalalignment = 'center')
    for i in xrange ( len ( images ) ) :
        ax0 = fig.add_subplot ( rows , cols ,( i +1) )        
        plt.setp ( ax0.get_xticklabels () , visible = False )
        plt.setp ( ax0.get_yticklabels () , visible = False )
        if len ( sptitles ) == len ( images ) :
            plt.title ( " % s #% s " % ( sptitle , str ( sptitles [ i ]) ))
        else :
            plt.title ( " % s #% d " % ( sptitle , ( i +1) ))
        plt.imshow ( np.asarray ( images [ i ]) , cmap = colormap )
        if filename is None :
            plt.show ()
        else :
            fig.savefig ( filename )




import matplotlib . cm as cm
# turn the first ( at most ) 16 eigenvectors into grayscale
# images ( note : eigenvectors are stored by column !)








def main():
    sys.path.append ( " .. " )
# import numpy and matplotlib colormaps
# read images
    [X , y ] = read_images ( "/home/pradeep/workspace/facerec/orl_faces/" )
	# perform a full pca
    [D , W , mu ] = pca ( asRowMatrix ( X ) , y )
    E = []
    for i in xrange ( min ( len ( X ) , 16) ) :
        e = W [: , i ]. reshape ( X [0]. shape )
        E.append ( normalize (e ,0 ,255) )
	# plot them and store the plot to " p y t h o n _ e i g e n f a c e s . pdf "
        subplot ( title = "Eigenfaces AT & T Facedatabase " , images =E , rows =4 , cols =4 , sptitle = " Eigenface " , colormap = cm.jet , filename = "python_pca_eignfaces.pdf" )


if __name__ == "__main__":
    main()

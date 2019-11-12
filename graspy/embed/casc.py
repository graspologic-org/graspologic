# casc
#' Covariate Assisted Spectral Clustering
#' 
#' @param adjMat An adjacency matrix
#' @param covMat A covariate matrix
#' @param nBlocks The number of clusters
#' @param nPoints Number of iterations to find the optimal tuning
#' parameter.
#' @param method The form of the adjacency matrix to be used.
#' @param rowNorm True if row normalization should be
#' done before running kmeans.
#' @param center A boolean indicating if the covariate matrix columns
#' should be centered.
#' @param verbose A boolean indicating if casc output should include eigendecomposition.
#' @param assortative A boolean indicating if the assortative version of casc should be used.
#' @param randStarts Number of random restarts for kmeans.
#'
#' @export
#' @return A list with node cluster assignments, the
#' the value of the tuning parameter used, the within
#' cluster sum of squares, and the eigengap.
#'
#' @keywords spectral clustering
import numpy as np
from sklearn import preprocessing
from numpy import linalg as la
from sklearn.cluster import KMeans
import scipy.sparse as sp
import warnings

def casc(adjMat, covMat, nBlocks, nPoints = 100,method = "regLaplacian", rowNorm = False,center = False,assortative = False):
	adjMat = getGraphMatrix(adjMat, method)
	covMat = preprocessing.scale(covMat,with_mean=center)
	res    = getCascClusters(adjMat, covMat, nBlocks, nPoints,rowNorm, assortative)
	return res

def getCascClusters(graphMat, covariates, nBlocks,nPoints, rowNorm, assortative):
	rangehTuning = getTuningRange(graphMat, covariates, nBlocks, assortative)
	hTuningSeq = np.linspace(rangehTuning[0], rangehTuning[1],nPoints)
	wcssVec = []
	gapVec = []
	orthoX = []
	orthoL = []
	for i in range(nPoints):
		cascResults = getCascResults(graphMat, covariates, hTuningSeq[i],nBlocks, rowNorm, assortative)
		orthoL.append(cascResults['orthoL'])
		orthoX.append(cascResults['orthoX'])
		wcssVec.append(cascResults['wcss'])
		gapVec.append(cascResults['gapVec'])
	hOpt = hTuningSeq[wcssVec.index(min(wcssVec))]
	hOptResults = getCascResults(graphMat, covariates, hOpt, nBlocks, rowNorm,assortative)
	return {'cluster':hOptResults['cluster'],'h':hOpt,'wcss':hOptResults['wcss'],'eigenGap':hOptResults['gapVec'],'cascSvd':hOptResults['cascSvd']}

def getCascResults(graphMat, covariates, hTuningParam,nBlocks, rowNorm,assortative):
	cascSvd = getCascSvd(graphMat, covariates, hTuningParam, nBlocks, assortative)
	ortho = getOrtho(graphMat, covariates, cascSvd['eVec'], cascSvd['eVal'],hTuningParam, nBlocks)
	if rowNorm:
		cascSvd_tmp = cascSvd['eVec']/np.sqrt(sum(cascSvd['eVec']**2))
	else:
		cascSvd_tmp = cascSvd['eVec']
	kmeansResults=KMeans(n_clusters=nBlocks).fit(cascSvd_tmp)
	return {'orthoL':ortho[0],'orthoX':ortho[1],'wcss':kmeansResults.inertia_,'cluster':kmeansResults.labels_,'gapVec':cascSvd['eVal'][nBlocks-1]-cascSvd['eVal'][nBlocks],'cascSvd':cascSvd_tmp}

def getOrtho(graphMat, covariates, cascSvdEVec, cascSvdEVal,h, nBlocks):
	orthoL=cascSvdEVec[:, (nBlocks-1)].transpose().dot(graphMat).dot(cascSvdEVec[:,(nBlocks-1)]).transpose()/cascSvdEVal[(nBlocks-1)]
	orthoX=h*(cascSvdEVec[:, (nBlocks-1)].transpose().dot(covariates).dot(covariates.transpose()).dot(cascSvdEVec[:,(nBlocks-1)])/cascSvdEVal[(nBlocks-1)])
	return [orthoL/(orthoL + orthoX),orthoX/(orthoL + orthoX)]

def getCascSvd(graphMat, covariates, hTuningParam, nBlocks, assortative):
	if assortative:
		def matmult(x, y):
			if len(x.shape)<2:
				res=(np.dot(graphMat,x) + np.dot(hTuningParam * covariates,np.dot(covariates.transpose(), x))).transpose()
			else:
				res=np.dot(graphMat,y) +  np.dot(hTuningParam * covariates, np.dot(covariates.transpose(), y))
			return res
	else:
		def matmult(x, y):
			if len(x.shape)<2:
				res=(np.dot(graphMat,np.dot(graphMat,x)) + np.dot(hTuningParam * covariates, np.dot(covariates.transpose(), x))).transpose()
			else:
				res=np.dot(graphMat,np.dot(graphMat,y)) + np.dot(hTuningParam * covariates,np.dot(covariates.transpose(), y))
			return res
	S=irlb(graphMat,nBlocks + 1,mult=matmult)
	eVec = S[0][:, 0:nBlocks]
	eVal = S[1][0:(nBlocks+1)]
	eVecKPlus = S[2][:,0:(nBlocks+1)]
	return {'eVec':eVec,'eVal':eVal,'eVecKPlus':eVecKPlus}

def getTuningRange(graphMatrix, covariates, nBlocks,assortative):
	nCov = covariates.shape[1]
	u,d,v=la.svd(covariates,full_matrices=False)
	min_tmp = np.min([nCov,nBlocks])
	singValCov = d[0:min_tmp]
	if assortative:
		u1,d1,v1 = la.svd(graphMatrix,full_matrices=False)
		tmp1 = nBlocks + 1
		eigenValGraph = d1[0:tmp1]
		if nCov > nBlocks:
			hmax = eigenValGraph[0]/(singValCov[nBlocks-1]**2 - singValCov[nBlocks]**2) 
		else:
			hmax = eigenValGraph[0]/singValCov[nCov-1]**2 
		hmin = (eigenValGraph[nBlocks-1] - eigenValGraph[nBlocks])/singValCov[0]**2
	else:
		u1,d1,v1 = la.svd(graphMatrix,full_matrices=False)
		tmp1 = nBlocks + 1
		eigenValGraph = d1[0:tmp1]**2
		if nCov > nBlocks :
			hmax = eigenValGraph[0]/(singValCov[nBlocks-1]**2 - singValCov[nBlocks]**2) 
		else:
			hmax = eigenValGraph[0]/singValCov[nCov-1]**2 
		hmin = (eigenValGraph[nBlocks-1] - eigenValGraph[nBlocks])/singValCov[0]**2
	return [hmax,hmin]

def getGraphMatrix(adjMat, method):
	if method == 'regLaplacian' :
		rSums = getRsum(adjMat)
		tau = np.mean(rSums)
		normMat = np.eye(len(rSums))* 1/np.sqrt(rSums + tau)
		return np.dot(normMat,adjMat,normMat)
	elif method == 'laplacian':
		rSums = getRsum(adjMat)
		normMat = np.eye(len(rSums))* 1/np.sqrt(rSums)
		return np.dot(normMat,adjMat,normMat)
	elif method == 'adjacency':
		return adjMat
	else:
		print("Error: method =", method, "Not valid. Method = ['regLaplacian','laplacian','adjacency']")
		return

def getRsum(Mat):
	Numrows=Mat.shape[0]
	rsums=[]
	for i in range(Numrows):
		rsums.append(np.sum(Mat[i]))
	return rsums


# Compute A.dot(x) if t is False,
#         A.transpose().dot(x)  otherwise.


# Simple utility function used to check linear dependencies during computation:
def irlb(A,n,tol=0.0001,maxit=50,mult="NULL"):
  if(mult == "NULL"):
    def mult(A,x,t=False):
      if(sp.issparse(A)):
        m = A.shape[0]
        n = A.shape[1]
        if(t):
          return(sp.csr_matrix(x).dot(A).transpose().todense().A[:,0])
        return(A.dot(sp.csr_matrix(x).transpose()).todense().A[:,0])
      if(t):
        return(x.dot(A))
      return(A.dot(x))
  def orthog(Y,X):
    dotY = mult(Y,X)
    return (Y - mult(X,dotY))
  def invcheck(x):
    eps2  = 2*np.finfo(np.float).eps
    if(x>eps2):
      x = 1/x
    else:
      x = 0
      warnings.warn("Ill-conditioning encountered, result accuracy may be poor")
    return(x)
  nu     = n
  m      = A.shape[0]
  n      = A.shape[1]
  if(min(m,n)<2):
    raise Exception("The input matrix must be at least 2x2.")
  m_b    = min((nu+20, 3*nu, n))  # Working dimension size
  mprod  = 0
  it     = 0
  j      = 0
  k      = nu
  smax   = 1
  sparse = sp.issparse(A)

  V  = np.zeros((n,m_b))
  W  = np.zeros((m,m_b))
  F  = np.zeros((n,1))
  B  = np.zeros((m_b,m_b))

  V[:,0]  = np.random.randn(n) # Initial vector
  V[:,0]  = V[:,0]/np.linalg.norm(V)

  while(it < maxit):
    if(it>0): j=k
    W[:,j] = mult(A,V[:,j])
    mprod+=1
    if(it>0):
      W[:,j] = orthog(W[:,j],W[:,0:j]) # NB W[:,0:j] selects columns 0,1,...,j-1
    s = np.linalg.norm(W[:,j])
    sinv = invcheck(s)
    W[:,j] = sinv*W[:,j]
    # Lanczos process
    while(j<m_b):
      F = mult(A,W[:,j])
      mprod+=1
      F = F - s*V[:,j]
      F = orthog(F,V[:,0:j+1])
      fn = np.linalg.norm(F)
      fninv= invcheck(fn)
      F  = fninv * F
      if(j<m_b-1):
        V[:,j+1] = F
        B[j,j] = s
        B[j,j+1] = fn 
        W[:,j+1] = mult(A,V[:,j+1])
        mprod+=1
        # One step of classical Gram-Schmidt...
        W[:,j+1] = W[:,j+1] - fn*W[:,j]
        # ...with full reorthogonalization
        W[:,j+1] = orthog(W[:,j+1],W[:,0:(j+1)])
        s = np.linalg.norm(W[:,j+1])
        sinv = invcheck(s) 
        W[:,j+1] = sinv * W[:,j+1]
      else:
        B[j,j] = s
      j+=1
    # End of Lanczos process
    S    = np.linalg.svd(B)
    R    = fn * S[0][m_b-1,:] # Residuals
    if(it<1):
      smax = S[1][0]  # Largest Ritz value
    else:
      smax = max((S[1][0],smax))

    conv = sum(np.abs(R[0:nu]) < tol*smax)
    if(conv < nu):  # Not coverged yet
      k = max(conv+nu,k)
      k = min(k,m_b-3)
    else:
      break
    # Update the Ritz vectors
    V[:,0:k] = V[:,0:m_b].dot(S[2].transpose()[:,0:k])
    V[:,k] = F 
    B = np.zeros((m_b,m_b))
    # Improve this! There must be better way to assign diagonal...
    for l in range(0,k):
      B[l,l] = S[1][l]
    B[0:k,k] = R[0:k]
    # Update the left approximate singular vectors
    W[:,0:k] = W[:,0:m_b].dot(S[0][:,0:k])
    it+=1

  U = W[:,0:m_b].dot(S[0][:,0:nu])
  V = V[:,0:m_b].dot(S[2].transpose()[:,0:nu])
  return((U,S[1][0:nu],V,it,mprod))

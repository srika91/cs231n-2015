import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  
  #x=x.reshape(x.shape[0],w.shape[0])
  
  out = x.reshape(x.shape[0],w.shape[0]).dot(w)+b
  
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx=dout.dot(w.T)
  dx=dx.reshape(x.shape)
  
  dw=x.reshape(x.shape[0],w.shape[0]).T.dot(dout)
  db=np.sum(dout,axis=0)
  #dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.maximum(0,x)
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx=(x>0)*dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  x_orig=x
  N,C,H,W=x.shape
  F,C,HH,WW=w.shape
  #print x.shape
  #print x
  pad=conv_param['pad']
  stride=conv_param['stride']
  Hf=1 + (H+ 2*pad-HH)/stride
  Wf=1 + (W+ 2*pad-WW)/stride
  #print Hf,Wf
  X_train=np.zeros((N,Hf*Wf,HH*WW*C))
  #print X_train.shape
  x=np.lib.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant',constant_values=0)
  #print x
  #print x.shape
  out=np.zeros((N,F,Hf,Wf))
  for k in range(0,N):
      for weight in range(0,F):
          for i in range(0,Hf):
              for j in range(0,Wf):
                  out[k,weight,i,j]=np.sum(x[k,:,stride*i:HH+stride*i,stride*j:WW+stride*j]*w[weight,:,:,:])+b[weight]


  '''
  count=0
  for k in range(0,N):
        count=0 
        for i in range(0,Hf):
              for j in range(0,Wf):
                  #print i*stride,WW+i*stride,j*stride,HH+j*stride          
                  X_train[k,count]=x[k,:,(i*stride):(WW+i*stride),(j*stride):(HH+j*stride)].flatten()
                  count=count+1
                  
  #print X_train.shape
  weights=np.zeros((F,C*HH*WW))
  out_temp=np.zeros((N,F,Hf*Wf))                
  for i in range(0,F):
        weights[i]=w[i].flatten()
  
  #print count
  out=np.zeros((N,F,Hf,Wf))
  
  for k in range(0,N):
       out_temp[k]=(X_train[k].dot(weights.T)+b.reshape(1,-1)).T

  '''

  '''
  for k in range(0,N):
        for i in range(0,F):
              for j in range(0,count): 
                  out_temp[k,i,j]=np.sum(X_train[k,j]*weights[i])+b[i]
  '''
  #print out_temp.shape
  #print out.shape
  #out=out_temp.reshape(N,F,Hf,Wf)                  
  #out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x_orig, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  
  x,w,b,conv_param=cache[0],cache[1],cache[2],cache[3]
  N,C,H,W=x.shape
  F,C,HH,WW=w.shape
  pad=cache[3]['pad']
  stride=cache[3]['stride']
  #print stride,pad
  #dout_temp=dout.reshape(N,F,Hf*Wf)
  Hf=1 + (H+ 2*pad-HH)/stride
  Wf=1 + (W+ 2*pad-WW)/stride
  #print Hf,Wf
  x_pad=np.lib.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant',constant_values=0)
  dx=np.zeros(x_pad.shape)
  print dx.shape
  dw=np.zeros(w.shape)
  db=np.zeros(b.shape)
  print dx.shape
  for k in range(0,N):
      for weights in range(0,F):
          for i in range(0,Hf):
              for j in range(0,Wf):
                  dw[weights]+=x_pad[k,:,i*stride:HH+i*stride,j*stride:WW+j*stride]*dout[k,weights,i,j]
                  dx[k,:,i*stride:HH+i*stride,j*stride:WW+j*stride]+=w[weights]*dout[k,weights,i,j]
  #If a value(position) is in many filter window will have its incoming gradients added in  dx
  
  dx=dx[:,:,1:-1,1:-1]
  print dx.shape
  dout_temp=dout.reshape(N,F,Hf*Wf)
  db=np.sum(np.sum(dout_temp,axis=0),axis=1)
  '''
  #print dout.shape
  dout_temp=dout.reshape(N,F,Hf*Wf)
  #print Hf,Wf
  X_train=np.zeros((N,Hf*Wf,HH*WW*C))
  #print X_train.shape
  
  #print x
  #print x.shape
  count=0
  for k in range(0,N):
        count=0 
        for i in range(0,Hf):
              for j in range(0,Wf):
                  #print i*stride,WW+i*stride,j*stride,HH+j*stride          
                  X_train[k,count]=x[k,:,(i*stride):(WW+i*stride),(j*stride):(HH+j*stride)].flatten()
                  count=count+1
  weights=np.zeros((F,C*HH*WW))
  #out_temp=np.zeros((N,F,Hf*Wf))                
  for i in range(0,F):
        weights[i]=w[i].flatten()
  
  dx, dw, db = np.zeros(x.shape), np.zeros(w.shape), np.zeros(b.shape)
  print dout_temp[0].shape,X_train[0].shape 
  dw=dout_temp[0].dot(X_train[0])
  print weights.T.shape,dout_temp[0].shape
  for k in range(0,N):
       dx[k]=dout_temp[k].T.dot(weights.T)
  
  db=np.sum(np.sum(dout_temp,axis=0),axis=1)
  '''
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  
  N,C,H,W=x.shape
  height=pool_param['pool_height']
  width=pool_param['pool_width']
  stride=pool_param['stride']   
  Wf=1 + (x.shape[3]-width)/stride 
  Hf=1 + (x.shape[2]-height)/stride
  
  out=np.zeros((N,C,Hf,Wf))
  for k in range(0,N):
      for channel in range(0,C):
          for i in range(0,Hf):
              for j in range(0,Wf):
                     out[k,channel,i,j]=np.max(x[k,channel,stride*i:stride*i+Hf,stride*j:stride*j+Wf])




  
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  
  ############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  x,pool_param=cache
  stride=pool_param['stride']
  height=pool_param['pool_height']
  width=pool_param['pool_width']
  Wf=1 + (x.shape[3]-width)/stride 
  Hf=1 + (x.shape[2]-height)/stride
  dx = np.zeros(x.shape)
  for k in range(0,x.shape[0]):
      for c in range(0,x.shape[1]):
          for i in range(0,Hf):
              for j in range(0,Wf):
                  temp=x[k,c,stride*i:stride*i + Hf,stride*j:stride*j+Wf]
                  dx[k,c,stride*i:stride*i + Hf,stride*j:stride*j+Wf]+=(temp==np.max(temp))*dout[k,c,i,j]
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  print probs.shape
  N = x.shape[0]
  print N
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx


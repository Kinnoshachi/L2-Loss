import numpy as np
from matplotlib import pyplot as plt

def f(x,w,b): 
    # x is a N-by-D numpy array
    # w is a D dimensional numpy array
    # b is a scalar
    # Should return a N dimensional numpy array
    y=sigmoid(np.dot(x,w)+b)
    
    return y



def sigmoid(t): 
    # t is a N dimensional numpy array
    # Should return a N dimensional numpy array
    return 1.0/(1.0+ np.nan_to_num(np.exp(-t)))



def l2loss(x,y,w,b): 
    # x is a N-by-D numpy array
    # y is a N dimensional numpy array
    # w is a D dimensional numpy array
    # b is a scalar
    # Should return three items: (i) the L2 loss which is a scalar, (ii) the gradient with respect to w, (iii) the gradient with respect to b
    fn=f(x,w,b)  
    loss = np.sum((y-fn)**2)
    dw = -2*np.sum(np.dot(((y-fn)*(1-fn)*fn),x)) #Partial derivative with respect to w
    db = -2*np.sum((y-fn)*(1-fn)*fn)            #Partial derivative with respect to b
    return [loss,dw,db]


def minimize_l2loss(x,y,w,b, num_iters=1000, eta=0.001): 
    # x is a N-by-D numpy array
    # y is a N dimensional numpy array
    # w is a D dimensional numpy array
    # b is a scalar
    # num_iters (optional) is number of iterations
    # eta (optional) is the stepsize for the gradient descent
    # Should return the final values of w and b
    losses = np.zeros(num_iters)
    for i in range(num_iters):
        loss,dw,db = l2loss(x,y,w,b)
        w = w - eta*dw  
        b = b - eta*db
        losses[i]=loss
    return w,b

    
# ROC and recall-precision functions here
def evaluate(scores,labels):
    
    thresholds = np.linspace(1,0,100) #generate 100 threshold points between 0 and 1

    ROC = np.zeros((len(thresholds),2))
    PRC = np.zeros((len(thresholds),2))
    for i in range(len(thresholds)):
        T = thresholds[i]
        
        # for current threshold.
        TP = np.logical_and( scores > T, labels==1 ).sum()  #True Positives
        TN = np.logical_and( scores <=T, labels==0 ).sum()  #True negatives
        FP = np.logical_and( scores > T, labels==0 ).sum()  #False Postives
        FN = np.logical_and( scores <=T, labels==1 ).sum()  #False Negaivtes
        
        
        # Compute false positive rate for current threshold.
        FPR = FP / float(FP + TN)
        ROC[i,0] = FPR
    
        # Compute true  positive rate for current threshold.
        TPR = TP / float(TP + FN)
        ROC[i,1] = TPR
        
       
        # Compute precision for current threshold.
        precision = (TP / float(TP + FP) if float(TP + FP)>0 else 1)
        PRC[i,0] = precision
    
        # Compute recall for current threshold.
        recall = (TP / float(TP + FN) if float(TP + FN) >0 else 1)
        PRC[i,1] = recall
    
    # Plot the ROC curve.
       
    
    #Calculate the Area Under the curve for ROC using the trapezoidal method  
    AUC_ROC = 0.
    for i in range(99):
        AUC_ROC += (ROC[i+1,0]-ROC[i,0]) * (ROC[i+1,1]+ROC[i,1])
    AUC_ROC *= 0.5
    
    #Calculate the Area Under the curve for PRC 
    AUC_PRC = 0.
    for i in range(99):
        AUC_PRC += (PRC[i+1,0]+PRC[i,0]) * (PRC[i+1,1]-PRC[i,1])  
    AUC_PRC *= 0.5
    
    
    
    #plot ROC curve
    f, (roc,prc) = plt.subplots(1,2)
    roc.plot(ROC[:,0], ROC[:,1], lw=2)
    roc.set_title('ROC curve, AUC = %.4f'%AUC_ROC)
    roc.set_xlim(-0.1,1.1)
    roc.set_ylim(-0.1,1.1)
    roc.plot([0,1],[0,1],'k--')
    roc.set_xlabel('$FPR(T)$')
    roc.set_ylabel('$TPR(T)$')
    roc.grid()        
    
    
    #plot PRC curve
    prc.plot(PRC[:,0], PRC[:,1], lw=2)
    prc.set_title('PRC curve, AUC = %.4f'%AUC_PRC)
    prc.set_xlim(-0.1,1.1)
    prc.set_ylim(-0.1,1.1)
    prc.set_xlabel('Recall')
    prc.set_ylabel('Precision')
    prc.grid() 
    
    plt.show()
        
    return AUC_ROC,AUC_PRC
            

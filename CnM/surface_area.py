import numpy as np
from scipy import signal



filter=np.array([1, 2, 4, 8, 16,32,64,128]).reshape((2,2,2))

image=np.zeros(shape=(40,40,80))+1

convoluted=signal.convolve(image,filter).astype('uint8')

arr=np.array([1, 2, 4, 8, 16,32,64,128])

cases=np.zeros(shape=image.shape=(,14))

weights=np.array([0.0,0.636,0.669,1.272,1.272,0.554,1.305,1.908,0.927,0.421,1.573,1.338,2.544,1.190])
cases[...,0]= np.isin(convoluted,np.array([0, 255])+1)*weights[0]

cases[...,1]=np.isin(convoluted,np.array([1, 2, 4, 8, 16, 32, 64, 128, 254, 253, 251, 247, 239, 223, 191, 127])+1)*weights[1]
cases[...,2]=np.isin(convoluted,np.array([3, 5, 17, 10, 34, 12, 68, 136, 48, 160, 192, 80, 252, 250, 238, 245, 221, 243, 187, 119, 207, 95, 63, 175])+1)*weights[2]
cases[...,3]=np.isin(convoluted,np.array([6, 9, 144, 96, 33, 18, 132, 72, 65, 20, 40, 130, 249, 246, 111, 159, 222, 237, 123, 183, 190, 235, 215, 125])+1)*weights[3]
cases[...,4]=np.isin(convoluted,np.array([129, 36, 66, 24, 126, 219, 189, 231])+1)*weights[4]
cases[...,5]=np.isin(convoluted,np.array([7,14,11,13,112,208,224,176,19,50,49,35,76,200,196,140,21,84,81,69,42,168,138,162,248,241,244,242,143,47,31,79,236,205,206,220,179,55,59,115,234,171,174,186,213,87,117,93])+1)*weights[5]
cases[...,6]=np.isin(convoluted,np.array([37,133,67,131,25,145,26,74,38,98,28,44,70,100,137,152,52,56,82,88,161,164,193,194,218,122,188,124,230,110,229,181,217,157,227,211,185,155,118,103,203,199,173,167,94,91,62,61)+1)*weights[6]
cases[...,7]=np.isin(convoluted,np.array([41, 73, 22, 134, 146, 148, 97, 104, 214, 182, 233, 121, 109, 107, 158, 151])+1)*weights[7]
case_9=np.array([15 240 51 204 85 170])+1
case_10=np.array([23 142 43 77 113 212 178 232])+1
case_11=np.array([39 71 27 139 29 141 46 78 114 116 177 184 209 216 226 228 53 202 58 197 83 172 92 163])+1
case_12=np.array([153 102 165 90 195 60])+1
case_13=np.array([105 150])+1
case_14=np.array([135 75 45 30 120 180 210 225 99 147 54 57 108 156 201 198 149 101 86 89 106 169 154 166])+1




case_1=np.isin(convoluted,arr)*
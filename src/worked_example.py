import numpy as np
from PIL import Image
import esvcp_fixed as E

cov=np.array(Image.open('data/cover/1.pgm'))
H=E.entropy_map(cov & np.uint8(0xF8),7)
# pick a 4x4 block that straddles the threshold
best=None
for i in [414]:
    for j in [462]:
        b=H[i:i+4,j:j+4]
        n=(b>=0.2).sum()
        if True:
            best=(i,j); break
    if best: break
i,j=best
blk=cov[i:i+4,j:j+4]; hb=H[i:i+4,j:j+4]
print(f"block at rows {i}-{i+3}, cols {j}-{j+3} of BOSSBase image 1\n")
print("pixel values:"); print(blk)
print("\nlocal entropy H~ (on planes B3-B7):")
print(np.round(hb,2))
elig=(hb>=0.2)
print("\neligible (H~ >= 0.20):"); print(elig.astype(int))
print(f"eligible count = {elig.sum()} of 16")

perm=E.logistic_map_perm(0.4567,3.9999,16)
print("\nlogistic permutation over the 16 raster indices:")
print(perm)
flat=elig.ravel()
pv=perm[flat[perm]]
print("\nfiltered to eligible set (embedding order):")
print(pv)
L=len(pv); n0,n1,n2=E._slices(L)
print(f"\nplane allocation: B0 {n0}, B1 {n1}, B2 {n2}  (60/30/10 of {L})")
print(" B0 indices:",pv[:n0])
print(" B1 indices:",pv[n0:n0+n1])
print(" B2 indices:",pv[n0+n1:])

bits=np.array([1,0,1,1,0,0,1,0,1,1,0,1][:L],dtype=np.uint8)
print("\npayload bits:",bits)
out=blk.ravel().copy()
def setbit(v,p,b):
    m=np.uint8(1<<p); return (v & ~m)|(b.astype(np.uint8)<<p)
out[pv[:n0]]=setbit(out[pv[:n0]],0,bits[:n0])
out[pv[n0:n0+n1]]=setbit(out[pv[n0:n0+n1]],1,bits[n0:n0+n1])
out[pv[n0+n1:]]=setbit(out[pv[n0+n1:]],2,bits[n0+n1:])
print("\nstego block:"); print(out.reshape(4,4))
print("\nchanged pixels:",(out!=blk.ravel()).sum(),"of 16; max |delta| =",int(np.abs(out.astype(int)-blk.ravel().astype(int)).max()))
# extraction
H2=E.entropy_map(out.reshape(4,4) & np.uint8(0xF8),7)
rec=np.empty(L,np.uint8)
rec[:n0]=(out[pv[:n0]]>>0)&1
rec[n0:n0+n1]=(out[pv[n0:n0+n1]]>>1)&1
rec[n0+n1:]=(out[pv[n0+n1:]]>>2)&1
print("\nrecovered bits:",rec,"  match:",bool((rec==bits).all()))

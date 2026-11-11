import numpy as np, sys
from itertools import combinations
from scipy.sparse.csgraph import minimum_spanning_tree as mst
from scipy.spatial.distance import pdist, squareform
from tda import rips, stable_ranks
def brute(X):
    """Independent reduction with no truncation, on python sets."""
    n=len(X); D=squareform(pdist(X))
    ed=sorted(combinations(range(n),2),key=lambda e:(D[e],e)); rank={e:i for i,e in enumerate(ed)}
    tr=[]
    for a,b,c in combinations(range(n),3):
        es=sorted([rank[(a,b)],rank[(a,c)],rank[(b,c)]],reverse=True); tr.append(es)
    tr.sort()
    pivot={}; bars=[]
    for es in tr:
        col=set(es)
        while col and max(col) in pivot: col^=pivot[max(col)]
        if col:
            l=max(col); pivot[l]=col
            b_=D[ed[l]]; d_=D[ed[es[0]]]
            if d_>b_: bars.append(d_-b_)
    return np.sort(bars)
r=np.random.default_rng(0); worst=0; cnt=0
for trial in range(300):
    n=int(r.integers(4,22)); d=int(r.integers(1,6)); X=r.normal(size=(n,d))
    H0,H1,_=rips([X]); b=brute(X)
    a=np.sort(H1[0]); assert len(a)==len(b),(len(a),len(b),n,d)
    if len(a): worst=max(worst,np.abs(a-b).max()); cnt+=len(a)
    M=squareform(pdist(X)); assert np.allclose(np.sort(H0[0]),np.sort(mst(M).data))
print('300 random clouds match brute force: bars',cnt,'worst error',worst)
th=np.linspace(0,2*np.pi,40,endpoint=False); C=np.c_[np.cos(th),np.sin(th)]
_,h,_b=rips([C]); print('circle: longest bar',round(np.sort(h[0])[-1],3),'bars',len(h[0]),'expected',round(2*np.sin(14*np.pi/40)-2*np.sin(np.pi/40),3))
S=np.array([[0,0],[1,0],[0,1],[1,1.]]); _,h,b=rips([S]); print('square birth,death',b[0][0],b[0][1],'expected 1, 1.414')
T=np.array([[0,0],[1,0],[0,1.]]); print('three corners H1 bars',len(rips([T])[1][0]),'expected 0')
L=np.sort(r.normal(size=(30,1)),0); print('points on a line H1 bars',len(rips([L])[1][0]),'expected 0')
print('stable rank',stable_ranks([np.array([3.,1.,2.])],np.array([0,.5,1,1.5,2,2.5,3])))
import time; cl=[r.normal(size=(25,32)) for _ in range(33000)]; t=time.time(); rips(cl); print('33000 clouds of 25 pts',round(time.time()-t,2),'s')
cl=[r.normal(size=(100,3)) for _ in range(200)]; t=time.time(); rips(cl); print('200 clouds of 100 pts',round(time.time()-t,2),'s')
# Distance-matrix entry point agrees with the coordinate entry point
from tda import rips_matrix, cubical
for t in range(100):
    X=r.normal(size=(int(r.integers(4,25)),3)); H0,H1,B=rips([X]); h0,b,d=rips_matrix(squareform(pdist(X)))
    assert np.allclose(np.sort(H0[0]),np.sort(h0)) and np.allclose(np.sort(H1[0]),np.sort(d-b))
print('rips_matrix agrees on 100 clouds')
# Cubical against a brute-force level-set component count on random images
for t in range(100):
    I=r.normal(size=(9,7)); bars=cubical(I[None])[0]
    # total number of merges equals local minima minus one
    from scipy.ndimage import minimum_filter
    foot=np.array([[0,1,0],[1,1,1],[0,1,0]],bool)
    nmin=(I==minimum_filter(I,footprint=foot,mode='constant',cval=np.inf)).sum()
    assert len(bars)==nmin-1,(len(bars),nmin)
# Exact case: two wells of depth 1 and 2 separated by a saddle at 0
I=np.array([[0.,-1,0,-2,0]]); print('cubical two wells',cubical(I[None])[0],'expected [1.]')

# ordered_match with unit masses is exactly the spike distance
from tda import ordered_match
import sys as _s; _s.path.insert(0, '..')
from spike_distance import vp_distance_matrix
if __name__ == "__main__":
    T=[np.sort(r.uniform(0,10,r.integers(1,40))) for _ in range(30)]
    for q in (0.5,10,100):
        A=ordered_match([(t,np.ones(len(t))) for t in T],q); B=vp_distance_matrix(T,q,jobs=1)
        assert np.allclose(A,B),q
    print('ordered_match with unit masses equals spike distance')

import numpy as np
import scipy.sparse as sp
import copy

# x = np.arange(10)
# y = np.arange(10,20)

# print np.outer(x,y)

# s = sp.dok_matrix((10,1))
# t = sp.dok_matrix((10,1))
# for i in range(10):
# 	s[i,0] = i
# 	t[i,0] = i + 10

# r = sp.kron(t,s.T).T
# print r.todense()


class Test(object):
    """Test how getitem should work."""
    def __init__(self):
        super(Test, self).__init__()
        self.arg1 = np.zeros(100)
        self.arg2 = 15.0
        self.arg3 = range(100)

    def __getitem__(self,key):
        new = copy.copy(self)
        new.arg3 = self.arg3[key]
        return new

def main():
    t = Test()
    l = t[3:15]

    print t.arg3
    print l.arg3

    l.arg3[0] = 0

    print t.arg3
    print l.arg3

    l.arg2 = 5.0
    print t.arg2
    print l.arg2

    l.arg1[0] =  1000
    print t.arg1
    print l.arg1

if __name__ == '__main__':
    main()
        
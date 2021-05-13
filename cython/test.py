import gpuadder
import numpy as np
import numpy.testing as npt
import gpuadder

def test():
    arr = np.array(np.linspace(1,128,128), dtype=np.int32)
    adder = gpuadder.GPUAdder(arr)
    adder.increment()
    
    adder.retreive_inplace()
    results2 = adder.retreive()

    npt.assert_array_equal(arr, np.linspace(1,128,128)+1)
    npt.assert_array_equal(results2, np.linspace(1,128,128)+1)
    
    print(arr)
    print(results2)

if __name__ == '__main__':
    test()

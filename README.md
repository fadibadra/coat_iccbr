coat_iccbr: CoAT source code for ICCBR'22
=======================================

The CoAT algorithm computes a measure of compatibility 
between two similarity measures on the whole dataset, and uses it to predict 
the most plausible way to complete the description of new data.

**Dependencies**
- numpy 
- scikit-learn
- matplotlib

 **Usage**
- python iccbr.py complexity [Balance|Iris|Pima|Monks1|Monks2|Monks3|User|Voting|Wine]
- python iccbr.py coat [Balance|Iris|Pima|Monks1|Monks2|Monks3|User|Voting|Wine]
- python iccbr.py E1 Balance [run|draw] 
- python iccbr.py E2 Balance [generate|results|draw]
- python iccbr.py E3 [run|draw|plot]

**Paper**

Fadi Badra, Marie-Jeanne Lesot, Aman Barakat, and Christophe Marsala, _Theoretical and Experimental Study of a Complexity Measure for Analogical Transfer_

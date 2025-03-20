# Interpret Corners

After uncrossing the efficient distinguishers with the tree of tangles algorithm we often have the problem of the efficient distinguishers not being features we originally added to the feature system but instead corners, or even worse, corners of corners of corners... 

Interpreting these corners can be a challenge, especially since tracking which corners we intersected to obtain a certain efficient distinguisher leads to complicated and highly redundant descriptions.

The tangles_tot library provides some tools to make it easier to interpret these corners by reconstructing a, hopefully, more simple description of features, in terms of intersections, unions and complements of the features originally added to the feature system. 

## Features and Logic Terms

We translate the concept of features, intersection, union and complement into the language of logic. 

We can associate a feature $A \subseteq V$ with the statement $v \in A$. For simplicity we describe this statement
simply by $A$. The inverse feature $V \setminus A$ can therefore be described by the statement $\lnot A$. 
Similarly the intersection $A \cap B$ of two features $A, B \subseteq V$ can be described by $A \land B$ and 
similarly the union $A \cup B$ by $A \lor B$. 

In this way the logic terms give us an interpretation of the meaning behind the efficient distinguisher features. 

## Reconstructing Features

Let us start with a simple example showing how the tool can be used.


```python
import numpy as np
from tangles_tot.features import interpret_feature
from tangles.separations.system import FeatureSystem

#building a feature system feat_sys which contains two features: A and B
features = np.array(
    [
        [1, -1],
        [1, 1],
        [-1, 1],
        [-1, -1]
    ],
)
metadata = ["A", "B"]
feat_sys = FeatureSystem.with_array(features, metadata = metadata)
len(feat_sys)
```




    2




```python
corner_ids, corner_specifications = feat_sys.get_corners(0, 1) #adding two corners to feat_sys
# the index 3 corresponds to the infimum of (0, 1) and (1, 1)
a_and_b = (corner_ids[3], corner_specifications[3])
# the index 0 corresponds to the infimum of (0, -1) and (1, -1), the inverse of "a or b"
a_or_b = (corner_ids[0], -corner_specifications[0])
print("a and b", a_and_b)
print("a or b", a_or_b)
```

    a and b (np.int64(5), np.int8(1))
    a or b (np.int64(2), np.int8(-1))



```python
print("(0, 1)", interpret_feature((0, 1), feat_sys))
print("(0, -1)", interpret_feature((0, -1), feat_sys))
print("(1, 1)", interpret_feature((1, 1), feat_sys))
print("(1, -1)", interpret_feature((1, -1), feat_sys))
print("a and b", interpret_feature((a_and_b), feat_sys))
print("inverse of (a and b)", interpret_feature((a_and_b[0], -a_and_b[1]), feat_sys))
print("a or b", interpret_feature((a_or_b), feat_sys))
print("inverse of (a or b)", interpret_feature((a_or_b[0], -a_or_b[1]), feat_sys))
```

    (0, 1) A
    (0, -1) ¬A
    (1, 1) B
    (1, -1) ¬B
    a and b A ∧ B
    inverse of (a and b) ¬B ∨ ¬A
    a or b A ∨ B
    inverse of (a or b) ¬A ∧ ¬B


## Conditional Features

While already helpful the previous terms can still grow quite long in practical tree of tangles. To allow for some simplification we introduce another method. This method allows us to calculate conditional features. Let us show what we mean by an example.

Suppose we have a tree of tangles consisting of the three nested features $A \cap B \leq A \leq A \cup B$. Then we could take $A$ as a "baseline" and consider for $A \cap B$ what it does _in addition_ to $A$, which, translated to the language of logic means that we look for a statement which is equal to $A \cup B$ under the condition that we know that $A$ is true. 

Similarly for the other feature $A \cup B$ under the condition $A$ is not interesting, as this is always true. But for their inverses $V \setminus (A \cup B)$ under the condition $A$ is $\lnot B$.  


```python
print("a and b under condition a   ", interpret_feature(a_and_b, feat_sys, under_condition=[(0, 1)]))
print("a or b under condition a   ", interpret_feature(a_or_b, feat_sys, under_condition=[(0, 1)]))
print("not (a or b) under condition not a   ", interpret_feature((a_or_b[0], -a_or_b[1]), feat_sys, under_condition=[(0, -1)]))
```

    a and b under condition a    B
    a or b under condition a    true
    not (a or b) under condition not a    ¬B


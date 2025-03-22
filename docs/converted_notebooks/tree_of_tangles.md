# Tree of Tangles


```python
from tangles_tot._testing.feature_trees import three_star
from tangles_tot.plot import plot_tree_of_tangles
from tangles_tot.tree import TreeOfTangles

tree_of_tangles = TreeOfTangles(three_star())
plot_tree_of_tangles(tree_of_tangles)
```


    
![png](../docs/converted_notebooks/tree_of_tangles_files/tree_of_tangles_1_0.png)
    



```python
plot_tree_of_tangles(
    tree_of_tangles, feature_specification=tree_of_tangles.default_specification()
)
```


    
![png](../docs/converted_notebooks/tree_of_tangles_files/tree_of_tangles_2_0.png)
    



```python

```

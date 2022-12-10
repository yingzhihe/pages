We propose implementing the following for additional features beyond the minimum requirements:

1. Newton Raphson method root finder (multivariate Newton's method)

2. Local minima and maxima over a given interval

3. Reverse-mode AD

The first two above should not require any significant changes to our code base, since we can use all of the existing machinery for computing derivatives. For the Newton Raphson method, we can have an iterative method, much like we did in class for Newton's method, but rather than using the single-variate derivative, we use the Jacobian. For the local minima, maxima, and saddle points, we calculate the gradient at a starting point, then step along the gradient for a certain step size (which we will find using a step size calculating algorithm). We repeat this iteratively until we reach a point where the gradient is sufficiently close to 0. 

However, implementation of reverse-mode AD would require a significant amount of change to the way derivatives are computed. We would have to add entirely new system of differentiation. For example, we would have to create a new data structure in order to store the computational graph. In forward-mode, we did not have to do this by virtue of our system using arithmetic and elementary function on dual numbers, but with reverse-mode, we must store all the computed partials from our forward pass through the computational graph in order to perform the reverse path. Due to how large the changes would be, we would likely have to add reverse-mode AD as a new module. We would likely add a new class of node (not dual numbers) in order to store the computational graph and partials. 

An overview of how we would implement reverse-mode AD: as highlighted in our milestone 1 progress file, we would create (at least) two new classes: a `DiffNode` class and a `DiffGraph` class. The `DiffNode` class implements the node of a computational graph, specific to reverse-mode AD, and stores the numerical value computed at the node as well as the partial derivatives of any subsequent child nodes, which will ultimately be used to compute the chain rule. The `DiffGraph` class will represent the DAG (directed acyclic graph) that consists of all of the nodes within the computational graph that is computed by parsing a function.

We will implement logic such as a graph searcher so that we can reuse nodes within a computational graph instead of having to recompute values. This can be done with hashing, for example, by assigning a hash value to each node within the graph and then using a hash map. 

Once we have implemented the core functionality of reverse-mode AD, we can extend our existing `jacobian_at()` function for `Function` and `VectorFunction` objects with an optional keyword argument to specify whether or not the computation is to be done using forward-mode or reverse-mode AD. 
# coding=utf-8
# Copyright 2020 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Simple tracer to transform function specifications into Layers.

Instead of having to figure out complicated combinator expressions, one can
simply write symbolic applications of layer objects to variables in a simple
python syntax and have it traced and 'compiled' into a layer object built from
simple combinators.

The DSL supports:
 - making input tuples of variables
 - applying layer objects to single variables or variable tuples
 - unpacking the result of layer object application

Further documentation is in the 'symbolic' docstring.
"""

import collections
import functools
import inspect

from trax.layers import base
from trax.layers import combinators as cb


# Trace Construction
# -----------------------------------------------------------------------------

# Representation for expressions  lyr(*args)  and  val[idx]
ApplyExpr = collections.namedtuple('ApplyExpr', ['lyr', 'args'])
IndexExpr = collections.namedtuple('IndexExpr', ['idx', 'val'])
# Representation for equations   dst = lyr(*src)  and  dst = src[idx]
ApplyEqn = collections.namedtuple('ApplyEqn', ['lyr', 'src', 'dst'])
IndexEqn = collections.namedtuple('IndexEqn', ['idx', 'src', 'dst'])


class Tracer(object):
  """Simple Tracer for handling layer application, input and output tuples."""

  def __init__(self, expr, n=0):
    self.expr = expr
    self.n = n

  def __repr__(self):
    return str(self.expr)

  # symbolic output tuple unpacking
  def __iter__(self):
    if self.n == 0:
      raise ValueError('Tracer: only the output '
                       'of Layer applications can be unpacked.')
    for i in range(self.n):
      yield Tracer(IndexExpr(i, self.expr))


def apply_to_tracer(self, other):
  """Records base.Layer operating on Tracers or tuples of Tracers."""
  if isinstance(other, Tracer):
    args = (other.expr,)
  elif (isinstance(other, (tuple, list)) and
        all([isinstance(x, Tracer) for x in other])):
    args = tuple(elm.expr for elm in other)
  else:
    raise ValueError('Layers can only apply to tracers and '
                     'tuples of tracers during a symbolic trace.')
  if len(args) != self.n_in:
    raise ValueError('Tracer: Layer takes %d inputs,'
                     ' got %d.' %  (self.n_in, len(args)))
  return Tracer(ApplyExpr(self, args), self.n_out)
# Bind this application function to the matmul '@' operator of base.Layer:
base.Layer.__matmul__ = apply_to_tracer


# Traced expressions to SSA equation form
# -----------------------------------------------------------------------------
def traces_to_eqns(traces):
  r"""Combines multiple traces into a set of primitive equations.

  If we trace a function like the following:
    def fun(a, b, c):
      d = L1 @ (a, b)
      e, f = L2 @ (c, d)
      return d, e

  Our two output traces will hold expression trees:
    d:   a ---\              e:   a ---\
         b --- L1 --->            b --- L1 ---\
                                  c ---------- L2 ---\
                                  0 ----------------- [] --->

  We break these expressions up node by node into simple equations
  by assigning the same variable names to hash-identical subtrees:
    eqns: var1 = L1(a, b), var2 = L2(c, var1), var3 = var2[0]
    outputs: var1, var3

  Note: a subtree containing a layer 'L' is only identical to another
  subtree if it contains the _same_ python object 'L'.

  Args:
    traces: a set of Tracers holding expression traces.

  Returns:
    A set of deduplicated SSA style assignments from layer applications and
    tuple indexing.
  """
  traces = traces if isinstance(traces, (tuple, list)) else (traces,)
  symboltable = {}  # map from expression hashes to unique symbol names
  eqns = []  # list of trace-derived equations

  def getsymbol(expr):
    """Assign unique ids to expression subtrees."""
    # Unique input atoms are represented by simple strings.
    if isinstance(expr, str):
      return expr
    # Identical expression trees are identified by their hash.
    h = hash(expr)
    if h in symboltable:
      return symboltable[h]
    newsym = 'var{}'.format(len(symboltable))
    symboltable[h] = newsym
    return newsym

  # Collect output symbols, which are the roots of each traced expression tree.
  output_symbols = tuple(getsymbol(trace.expr) for trace in traces)

  # Recursively collect equations from nodes of the traced expression trees
  # rooted in each output variable trace.
  def node_to_eqn(expr):
    """Transform expression graph nodes into simple equations."""
    newsym = getsymbol(expr)
    if isinstance(expr, IndexExpr):
      eqns.append(IndexEqn(expr.idx, node_to_eqn(expr.val), newsym))
    elif isinstance(expr, ApplyExpr):
      eqns.append(
          ApplyEqn(expr.lyr, tuple(map(node_to_eqn, expr.args)), (newsym,)))
    return newsym

  for trace in traces:
    node_to_eqn(trace.expr)

  # Remove duplicate equations, taking care to otherwise
  # preserve original ordering for deterministic results.
  # TODO(levskaya): replace with normal dict once Py2 is EOL.
  eqns = list(collections.OrderedDict.fromkeys(tuple(eqns)))
  return eqns, output_symbols


# Symbolic Simplifications
# -----------------------------------------------------------------------------
def merge_output_tuples(eqns):
  """Combine all indexing eqns and rewrite apply eqns to use output tuples.

  Args:
    eqns: primitive equations representing separate output tuple members:
      var2 = var4[1]
      var1 = var4[0]
      var4 = layerA(var5, var6)

  Returns:
    simplified set of applications mapping inputs tuples to output tuples:
      var1, var2 = layerA(var5, var6)
  """
  # Gather all seen outputs of tuple variables, associate each
  # with the symbol for their source tuple and their tuple index.
  idx_eqns = [e for e in eqns if isinstance(e, IndexEqn)]
  output_tuples = dict([(eqn.src, {}) for eqn in idx_eqns])
  for eqn in idx_eqns:
    output_tuples[eqn.src][eqn.idx] = eqn.dst
  # Rewrite layer applications in terms of the collected output tuples.
  apply_eqns = [e for e in eqns if isinstance(e, ApplyEqn)]
  for i, eqn in enumerate(apply_eqns):
    # each application eqn only has a single output symbol at this stage,
    # if it's a symbol corresponding to a tuple, replace it by the set
    # of observed tuple outputs.
    if eqn.dst[0] in output_tuples:
      mapped_outputs = output_tuples[eqn.dst[0]]
      assert len(mapped_outputs) <= eqn.lyr.n_out
      out_vars = ['_%d'%j for j in range(eqn.lyr.n_out)]
      for idx, var in mapped_outputs.items():
        out_vars[idx] = var
      apply_eqns[i] = ApplyEqn(eqn.lyr, eqn.src, tuple(out_vars))
  return apply_eqns


def toposort(graph, start):
  """Standard topological sort of graph from start nodes.

  Vertices are represented by integer ids.

  Args:
    graph: graph represented by dictionary keyed by vertex id,
      each value being a list of the connected vertex ids.
    start: list of starting vertices' ids for initializing
      topological sort.

  Returns:
    list of vertex ids giving a topological sort of vertices.
  """
  seen, stack, order = set(), [], []
  q = start
  while q:
    v = q.pop()
    if v not in seen:
      seen.add(v)
      q.extend(graph[v])
      while stack and v not in graph[stack[-1]]:
        order.append(stack.pop())
      stack.append(v)
  return stack + order[::-1]


def evaluation_order_sort(eqns, outputs):
  """Sort eqns into evaluation order by topological sort.

  Args:
    eqns: list of ApplyEqns derived from dataflow traces.
    outputs: list of strings representing output symbols.

  Returns:
    list of ApplyEqns sorted into an evaluation order respecting
    dependencies among variables.
  """
  # Build a dependency graph between equations
  dependency_graph = {i: [] for i, _ in enumerate(eqns)}
  for i, eqn_a in enumerate(eqns):
    for j, eqn_b in enumerate(eqns):
      if set(eqn_a.src).intersection(set(eqn_b.dst)):
        dependency_graph[i].append(j)
  # Gather the equations emitting known output variables
  output_nodes = []
  for i, eqn in enumerate(eqns):
    if set(outputs).intersection(set(eqn.dst)):
      output_nodes.append(i)
  # Topological sort starting from outputs, the reverse is
  # then an evaluation ordering.
  topological_order = toposort(dependency_graph, output_nodes)
  return [eqns[i] for i in topological_order[::-1]]


# Layer object creation
# -----------------------------------------------------------------------------
def recombine(eqns, inputs, outputs):
  """Implement derived equations via layer-applications and combinators.

  Args:
    eqns: list of ApplyEqns derived from dataflow traces.
    inputs: list of strings representing input symbols
    outputs: list of strings representing output symbols

  Returns:
    Trax layer object that implements the given dataflow on provided layers.
  """
  stack = tuple(inputs)  # models the data stack
  layers = []  # output trax layers

  # Keep track of what variables are still needed after each
  # layer application so we can discard unnecessary variables
  # from the data stack.
  keepsets = [set(outputs)]
  for e in reversed(eqns):
    keepsets.append(keepsets[-1].union(e.src))
  keepsets = list(reversed(keepsets[:-1]))

  # For each layer application, rearrange the data stack to supply
  # its inputs, copying arguments needed later on.
  for eqn, keep in zip(eqns, keepsets):
    remainder = tuple(s for s in stack if s in keep)
    # only insert data-routing layer if needed:
    if stack != eqn.src + remainder:
      select_indices = [stack.index(var) for var in eqn.src + remainder]
      layers.append(cb.Select(select_indices, len(stack)))
    # stack now equals eqn.src + remainder
    layers.append(eqn.lyr)
    stack = eqn.dst + remainder
  # Finally, if needed, select out the final outputs from the data stack.
  if stack != tuple(outputs):
    layers.append(cb.Select([stack.index(var) for var in outputs], len(stack)))
  return cb.Serial(*layers)


def split_signature_parameters(fn):
  """Extract a function's positional and keyword arguments, ignoring varargs.

  Args:
    fn: a function

  Returns:
    A tuple of: a list of no-default positional arguments
     and a dict of the kwargs with provided defaults.
  """
  positional_kinds = {inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD}
  keyword_kinds = {inspect.Parameter.KEYWORD_ONLY,
                   inspect.Parameter.POSITIONAL_OR_KEYWORD}
  positionals, kwargs = [], {}
  for pname, pvar in inspect.signature(fn).parameters.items():
    if pvar.default == inspect._empty and pvar.kind in positional_kinds:  # pylint: disable=protected-access
      positionals.append(pname)
    elif pvar.default != inspect._empty and pvar.kind in keyword_kinds:  # pylint: disable=protected-access
      kwargs[pname] = pvar.default
  return positionals, kwargs


# The exported, user-facing API call.
# -----------------------------------------------------------------------------
def symbolic(fn):
  """Decorator to trace and combine layers using natural python notation.

  Instead of having to figure out complicated combinator expressions, one can
  simply write symbolic applications of layer objects to variables in a simple
  python syntax and have it traced and 'compiled' into low-level combinators.

  This decorator takes a simple function-based DSL description of layer
  combinations and produces a layer construction function that can optionally
  take keyword arguments to override configuration variables.

  The DSL supports:
  - applying layer objects to single variables or variable tuples
  - unpacking the result of layer object application
  - layer objects can also be created inside and used

  @tl.symbolic
  def new_trax_layer(a, b, c, config_var=True):
    d, e = layer_objectA @ (a, b)
    if config_var:
      f = layer_objectB @ c
    else:
      other_layer = some_other_layer_constructor()
      f = other_layer @ c
    g = tl.Serial(layer_objectC, layer_objectD) @ (d, e, f, g)
    return g, f, a

  NOTE: the functions provided can have two kinds of arguments:
  - positional: these name variables that will flow into the layer
  - keyword arguments: these are -configuration- variables that will
      not be traced, but can be given as kwargs to the layer constructor
      function that this decorator produces.

  The above creates a trax layer constructor that takes a single keyword
  argument `config_var` producing a trax layer that takes three array
  arguments and returns three arrays, e.g. we can call it like:

  layer = new_trax_layer()  # config_var = True
  or:
  tl.Serial(tl.Dense, new_trax_layer(config_var=False))

  Note: for python2 compatibility, the '<<' operator can be used in
  place of '@' as:  d, e = layer_objectA << (a, b)

  Args:
    fn: any python function following the above tracing conventions for
      describing dataflow between trax layers.

  Returns:
    Trax layer object implementing the dataflow between the trax layers
    used in the provided function.
  """
  fn_args, fn_kwargs = split_signature_parameters(fn)
  n_args = len(fn_args)
  if n_args == 0:
    raise ValueError('Must have named positional arguments to trace.')

  def traced_layer_constructor(*args, **kwargs):
    """Constructs trax layer."""
    # Check and handle arguments.
    if args:
      raise ValueError('Layer constructor takes no positional arguments.')
    extra_kwargs = set(kwargs).difference(set(fn_kwargs))
    if extra_kwargs:
      raise ValueError('Unknown layer constructor parameters: '
                       '%s' % extra_kwargs)
    fn_kwargs.update(kwargs)
    traced_fn = functools.partial(fn, **fn_kwargs)
    # Trace through positional arguments.
    tracers = [Tracer('in{}'.format(i)) for i in range(n_args)]
    returned_tracers = traced_fn(*tracers)
    # Transform traces back into ordered, simplified equations.
    inputs = tuple('in{}'.format(i) for i in range(n_args))
    eqns, outputs = traces_to_eqns(returned_tracers)
    eqns = merge_output_tuples(eqns)
    eqns = evaluation_order_sort(eqns, outputs)
    # Compose the traced layer DAG with combinators.
    return recombine(eqns, inputs, outputs)

  return traced_layer_constructor

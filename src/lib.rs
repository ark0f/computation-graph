use petgraph::visit::DfsEvent;
use std::cell::Cell;
use std::iter;

pub use petgraph::graph::NodeIndex;

/// Computation graph itself
///
/// After addition of new nodes or inputs you have to rebuild graph edges manually
/// using [`build_edges()`].
///
/// [`build_edges()`]: Engine::build_edges
#[derive(Default)]
pub struct ComputationGraph {
    inputs: Vec<f32>,
    graph: petgraph::Graph<EngineNode, ()>,
}

impl ComputationGraph {
    pub fn add_node<T: Node + 'static>(&mut self, node: T) -> NodeIndex {
        self.graph.add_node(EngineNode::Cache(CacheNode {
            inner: Box::new(node),
            cache: Cell::new(Cache::Invalid),
        }))
    }

    pub fn add_input(&mut self) -> Input {
        let input_idx = self.inputs.len();
        self.inputs.push(0.0);
        let node_idx = self.graph.add_node(EngineNode::Input(InputNode(input_idx)));
        Input {
            node_idx,
            input_idx,
        }
    }

    pub fn set_input(&mut self, input: Input, data: f32) {
        self.inputs[input.input_idx] = data;

        let mut invalid_nodes: Vec<NodeIndex> = vec![];
        petgraph::visit::depth_first_search(&self.graph, Some(input.node_idx), |event| {
            if let DfsEvent::Discover(n, _) = event {
                invalid_nodes.push(n);
            }
        });

        for idx in invalid_nodes {
            let weight = self.graph.node_weight_mut(idx);
            if let Some(EngineNode::Cache(node)) = weight {
                node.cache.set(Cache::Invalid);
            }
        }
    }

    fn build_edges_vec(&mut self, idx: NodeIndex) -> Vec<(NodeIndex, NodeIndex)> {
        let mut edges = vec![];
        let mut stack = vec![idx];
        while let Some(idx) = stack.pop() {
            let node = self.graph.node_weight(idx).unwrap();
            let deps = node.dependencies();
            stack.extend(deps.iter());

            let new_edges = deps.into_iter().zip(iter::repeat(idx));
            edges.extend(new_edges);
        }

        edges
    }

    pub fn build_edges(&mut self, idx: NodeIndex) {
        self.graph.clear_edges();
        let edges = self.build_edges_vec(idx);
        for (a, b) in edges {
            self.graph.update_edge(a, b, ());
        }
    }

    pub fn compute(&mut self, idx: NodeIndex) -> f32 {
        let node = self.graph.node_weight(idx).unwrap();
        node.compute(self)
    }
}

pub trait Node {
    fn dependencies(&self) -> Vec<NodeIndex>;

    fn compute(&self, engine: &ComputationGraph) -> f32;
}

impl Node for NodeIndex {
    fn dependencies(&self) -> Vec<NodeIndex> {
        vec![]
    }

    fn compute(&self, engine: &ComputationGraph) -> f32 {
        engine.graph.node_weight(*self).unwrap().compute(engine)
    }
}

type BoxedNode = Box<dyn Node>;

impl Node for BoxedNode {
    fn dependencies(&self) -> Vec<NodeIndex> {
        (**self).dependencies()
    }

    fn compute(&self, engine: &ComputationGraph) -> f32 {
        (**self).compute(engine)
    }
}

// such node is something like specialization to avoid InputNode caching
enum EngineNode {
    Cache(CacheNode<BoxedNode>),
    Input(InputNode),
}

impl Node for EngineNode {
    fn dependencies(&self) -> Vec<NodeIndex> {
        match self {
            EngineNode::Cache(cache) => cache.dependencies(),
            EngineNode::Input(input) => input.dependencies(),
        }
    }

    fn compute(&self, engine: &ComputationGraph) -> f32 {
        match self {
            EngineNode::Cache(cache) => cache.compute(engine),
            EngineNode::Input(input) => input.compute(engine),
        }
    }
}

struct CacheNode<T> {
    inner: T,
    cache: Cell<Cache>,
}

impl<T: Node> Node for CacheNode<T> {
    fn dependencies(&self) -> Vec<NodeIndex> {
        self.inner.dependencies()
    }

    fn compute(&self, engine: &ComputationGraph) -> f32 {
        match self.cache.get() {
            Cache::Valid(data) => data,
            Cache::Invalid => {
                let data = self.inner.compute(engine);
                self.cache.set(Cache::Valid(data));
                data
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum Cache {
    Valid(f32),
    Invalid,
}

#[derive(Debug, Copy, Clone)]
pub struct Input {
    pub node_idx: NodeIndex,
    input_idx: usize,
}

#[derive(Debug, Eq, PartialEq)]
struct InputNode(usize);

impl Node for InputNode {
    fn dependencies(&self) -> Vec<NodeIndex> {
        vec![]
    }

    fn compute(&self, engine: &ComputationGraph) -> f32 {
        engine.inputs[self.0]
    }
}

pub struct SumNode(pub NodeIndex, pub NodeIndex);

impl Node for SumNode {
    fn dependencies(&self) -> Vec<NodeIndex> {
        vec![self.0, self.1]
    }

    fn compute(&self, engine: &ComputationGraph) -> f32 {
        self.0.compute(engine) + self.1.compute(engine)
    }
}

pub struct MulNode(pub NodeIndex, pub NodeIndex);

impl Node for MulNode {
    fn dependencies(&self) -> Vec<NodeIndex> {
        vec![self.0, self.1]
    }

    fn compute(&self, engine: &ComputationGraph) -> f32 {
        self.0.compute(engine) * self.1.compute(engine)
    }
}

pub struct SinNode(pub NodeIndex);

impl Node for SinNode {
    fn dependencies(&self) -> Vec<NodeIndex> {
        vec![self.0]
    }

    fn compute(&self, engine: &ComputationGraph) -> f32 {
        self.0.compute(engine).sin()
    }
}

pub struct PowNode(pub NodeIndex, pub f32);

impl Node for PowNode {
    fn dependencies(&self) -> Vec<NodeIndex> {
        vec![self.0]
    }

    fn compute(&self, engine: &ComputationGraph) -> f32 {
        self.0.compute(engine).powf(self.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Expr {
        x1: Input,
        x2: Input,
        x3: Input,
        pow: NodeIndex,
        pow_sum: NodeIndex,
        sin: NodeIndex,
        mul: NodeIndex,
        output: NodeIndex,
    }

    // y = x1 + x2 * sin(x2 + pow(x3, 3))
    fn build_expr(engine: &mut ComputationGraph) -> Expr {
        let x1 = engine.add_input();
        let x2 = engine.add_input();
        let x3 = engine.add_input();

        let pow = engine.add_node(PowNode(x3.node_idx, 3.0));
        let pow_sum = engine.add_node(SumNode(x2.node_idx, pow));
        let sin = engine.add_node(SinNode(pow_sum));
        let mul = engine.add_node(MulNode(x2.node_idx, sin));
        let final_sum = engine.add_node(SumNode(x1.node_idx, mul));
        engine.build_edges(final_sum);

        Expr {
            x1,
            x2,
            x3,
            pow,
            pow_sum,
            sin,
            mul,
            output: final_sum,
        }
    }

    // round to decimal digits
    fn round(x: f32, precision: u32) -> f32 {
        let m = 10i32.pow(precision) as f32;
        (x * m).round() / m
    }

    #[test]
    fn compute_works() {
        let mut engine = ComputationGraph::default();
        let Expr {
            x1, x2, x3, output, ..
        } = build_expr(&mut engine);
        engine.set_input(x1, 1.0);
        engine.set_input(x2, 2.0);
        engine.set_input(x3, 3.0);
        let res = engine.compute(output);
        assert_eq!(round(res, 5), -0.32727);
    }

    #[test]
    fn cache_works() {
        fn assert_cache_is_valid<'a, T: Iterator<Item = &'a EngineNode>>(iter: T) {
            for node in iter {
                if let EngineNode::Cache(node) = node {
                    assert!(matches!(node.cache.get(), Cache::Valid(_)));
                }
            }
        }

        let mut engine = ComputationGraph::default();
        let Expr { x1, output, .. } = build_expr(&mut engine);

        engine.compute(output);

        // assert all nodes being cached
        assert_cache_is_valid(engine.graph.node_weights());

        // check only output node cache being invalidated
        engine.set_input(x1, 123.321);
        for idx in engine.graph.node_indices() {
            let node = engine.graph.node_weight(idx).unwrap();
            if let EngineNode::Cache(node) = node {
                if idx == output {
                    assert_eq!(node.cache.get(), Cache::Invalid);
                } else {
                    assert!(matches!(node.cache.get(), Cache::Valid(_)));
                }
            }
        }

        engine.compute(output);
        assert_cache_is_valid(engine.graph.node_weights());
    }

    #[test]
    #[rustfmt::skip]
    fn edges_built_correctly() {
        let mut engine = ComputationGraph::default();
        let Expr {
            x1,
            x2,
            x3,
            pow,
            pow_sum,
            sin,
            mul,
            output,
        } = build_expr(&mut engine);

        let edges = engine.build_edges_vec(output);
        assert_eq!(edges, vec![
            (x1.node_idx, output),
            (mul, output), (x2.node_idx, mul), 
            (sin, mul), (pow_sum, sin), (x2.node_idx, pow_sum),
            (pow, pow_sum), (x3.node_idx, pow) 
        ]);
    }
}

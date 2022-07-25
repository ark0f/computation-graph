use computation_graph::{ComputationGraph, MulNode, PowNode, SinNode, SumNode};

fn main() {
    let mut engine = ComputationGraph::default();

    let x1 = engine.add_input();
    let x2 = engine.add_input();
    let x3 = engine.add_input();

    let pow = engine.add_node(PowNode(x3.node_idx, 3.0));
    let pow_sum = engine.add_node(SumNode(x2.node_idx, pow));
    let sin = engine.add_node(SinNode(pow_sum));
    let mul = engine.add_node(MulNode(x2.node_idx, sin));
    let final_sum = engine.add_node(SumNode(x1.node_idx, mul));

    engine.set_input(x1, 1.0);
    engine.set_input(x2, 2.0);
    engine.set_input(x3, 3.0);
    let result = engine.compute(final_sum);
    println!("Graph output = {}", result);

    engine.set_input(x1, 2.0);
    engine.set_input(x2, 3.0);
    engine.set_input(x3, 4.0);
    let result = engine.compute(final_sum);
    println!("Graph output = {}", result);
}

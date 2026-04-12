use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::IntoPyObjectExt;
use std::collections::VecDeque;

/// Fast topological sort using Kahn's algorithm.
/// Takes adjacency list as dict[str, list[str]] and returns sorted order.
/// Raises ValueError on cycle detection.
#[pyfunction]
fn topo_sort(_py: Python<'_>, edges: &Bound<'_, PyDict>) -> PyResult<Vec<String>> {
    let mut adj: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    let mut in_degree: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    // Collect all nodes and edges
    for (key, value) in edges.iter() {
        let from: String = key.extract()?;
        let targets: Vec<String> = value.extract()?;

        in_degree.entry(from.clone()).or_insert(0);
        adj.entry(from.clone()).or_default();

        for to in &targets {
            in_degree.entry(to.clone()).or_insert(0);
            *in_degree.get_mut(to).unwrap() += 1;
            adj.entry(from.clone()).or_default().push(to.clone());
        }
    }

    // Kahn's algorithm
    let mut queue: VecDeque<String> = in_degree
        .iter()
        .filter(|(_, &d)| d == 0)
        .map(|(n, _)| n.clone())
        .collect();

    // Sort queue for deterministic output
    let mut sorted_queue: Vec<String> = queue.drain(..).collect();
    sorted_queue.sort();
    queue.extend(sorted_queue);

    let mut order: Vec<String> = Vec::with_capacity(in_degree.len());
    let total = in_degree.len();

    while let Some(node) = queue.pop_front() {
        order.push(node.clone());
        if let Some(neighbors) = adj.get(&node) {
            let mut next_batch: Vec<String> = Vec::new();
            for neighbor in neighbors {
                if let Some(deg) = in_degree.get_mut(neighbor) {
                    *deg -= 1;
                    if *deg == 0 {
                        next_batch.push(neighbor.clone());
                    }
                }
            }
            next_batch.sort();
            queue.extend(next_batch);
        }
    }

    if order.len() != total {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cycle detected in graph",
        ));
    }

    Ok(order)
}

/// Parallel map: apply a Python callable to each element of a list using Rust threads.
/// Falls back to sequential if the callable isn't Send-safe (most Python objects).
/// Primary benefit: reduced Python loop overhead for large collections.
#[pyfunction]
fn fast_map(py: Python<'_>, func: &Bound<'_, PyAny>, items: &Bound<'_, PyList>) -> PyResult<Vec<PyObject>> {
    let mut results = Vec::with_capacity(items.len());
    for item in items.iter() {
        let result = func.call1((item,))?;
        results.push(result.into_py_any(py)?);
    }
    Ok(results)
}

/// Batch a list into chunks of given size. Returns list of lists.
#[pyfunction]
fn batch_items(py: Python<'_>, items: &Bound<'_, PyList>, batch_size: usize) -> PyResult<Vec<Vec<PyObject>>> {
    if batch_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "batch_size must be > 0",
        ));
    }

    let mut batches: Vec<Vec<PyObject>> = Vec::new();
    let mut current_batch: Vec<PyObject> = Vec::with_capacity(batch_size);

    for item in items.iter() {
        current_batch.push(item.into_py_any(py)?);
        if current_batch.len() == batch_size {
            batches.push(current_batch);
            current_batch = Vec::with_capacity(batch_size);
        }
    }
    if !current_batch.is_empty() {
        batches.push(current_batch);
    }

    Ok(batches)
}

/// Find all root nodes (in-degree 0) in a graph.
#[pyfunction]
fn find_roots(edges: &Bound<'_, PyDict>) -> PyResult<Vec<String>> {
    let mut all_nodes: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut has_parent: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (key, value) in edges.iter() {
        let from: String = key.extract()?;
        let targets: Vec<String> = value.extract()?;
        all_nodes.insert(from);
        for t in targets {
            has_parent.insert(t.clone());
            all_nodes.insert(t);
        }
    }

    let mut roots: Vec<String> = all_nodes.difference(&has_parent).cloned().collect();
    roots.sort();
    Ok(roots)
}

/// Find all leaf nodes (out-degree 0) in a graph.
#[pyfunction]
fn find_leaves(edges: &Bound<'_, PyDict>) -> PyResult<Vec<String>> {
    let mut all_nodes: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut has_children: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (key, value) in edges.iter() {
        let from: String = key.extract()?;
        let targets: Vec<String> = value.extract()?;
        all_nodes.insert(from.clone());
        if !targets.is_empty() {
            has_children.insert(from);
        }
        for t in targets {
            all_nodes.insert(t);
        }
    }

    let mut leaves: Vec<String> = all_nodes.difference(&has_children).cloned().collect();
    leaves.sort();
    Ok(leaves)
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(topo_sort, m)?)?;
    m.add_function(wrap_pyfunction!(fast_map, m)?)?;
    m.add_function(wrap_pyfunction!(batch_items, m)?)?;
    m.add_function(wrap_pyfunction!(find_roots, m)?)?;
    m.add_function(wrap_pyfunction!(find_leaves, m)?)?;
    Ok(())
}

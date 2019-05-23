# -*- coding: utf-8 -*-
#/usr/bin/python3
import tensorflow as tf
import keras
from keras import backend as K
from tensorflow.tools.graph_transforms import TransformGraph

# save model to pb ====================
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    Args:
        session : The TensorFlow session to be frozen.
        keep_var_names : A list of variable names that should not be frozen,\
                        or None to freeze all the variables in the graph.
        output_names : Names of the relevant graph outputs.
        clear_devices : Remove the device directives from the graph for better portability.
    
    Returns:
        The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                                              output_names, freeze_var_names)
        return frozen_graph

def freeze_keras(sess, input_node_names, output_node_names, optimize = False, quantize=False, clear_devices = True, add_tf_global = False):

    if add_tf_global:
        output_node_names += [v.op.name for v in tf.global_variables()]

    graphDef = sess.graph.as_graph_def()

    if clear_devices:
        for node in graphDef.node:
            node.device = ""

    if optimize:
        transforms = [
                         "merge_duplicate_nodes",
                         "strip_unused_nodes",
                         "fold_constants(ignore_errors=true)",
                         "fold_batch_norms",
                         "fold_old_batch_norms",
                         "sort_by_execution_order"
                        ]
        graphDef = TransformGraph(graphDef, [],
                                               output_node_names,
                                               transforms)

    if quantize:
        transforms = ["quantize_weights", "quantize_nodes"]
        graphDef = TransformGraph(graphDef, [],
                                               output_node_names,
                                               transforms)

    constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
                                                                            sess,
                                                                            graphDef,
                                                                            output_node_names)

    return constant_graph
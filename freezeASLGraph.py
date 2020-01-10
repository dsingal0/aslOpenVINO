import tensorflow
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
import glob
# Clear any previous session.
tensorflow.keras.backend.clear_session()

save_pb_dir = './model'
model_fname = glob.glob("aslClassifier*")[0]
def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tensorflow.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tensorflow.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

# This line must be executed before loading Keras model.
tensorflow.keras.backend.set_learning_phase(0) 

model = load_model(model_fname)

session = tensorflow.keras.backend.get_session()

INPUT_NODE = [t.op.name for t in model.inputs]
OUTPUT_NODE = [t.op.name for t in model.outputs]
print(INPUT_NODE, OUTPUT_NODE)
frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)

import argparse
import tensorflow as tf
import tensorflow.keras as tfk

from distribution import GaussianDynamicDiagVar
from scod import SCOD
from sketching import GaussianSketchOp
from dnn.data import VODataPipe
from dnn.model import DeepPose
from utils.params import ParamDict
from utils.tf_utils import set_tf_memory_growth

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params", type=str, default="params.py",
                        help="path to load the parameter file")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="path to load model from")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="the batch size used for running SCOD")
    parser.add_argument("--repeat", type=int, default=4,
                        help="train dataset repeat count for sampling")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="path to store output model with uncertainty")
    parser.add_argument("--debug", action="store_true",
                        help="specify this to limit the amount of data to be processed")

    return parser.parse_args()

if __name__ == "__main__":
    set_tf_memory_growth()
    args = parse_args()
    p = ParamDict.from_file(args.params)

    # load trained model
    model = DeepPose(p.model).build_model()
    model.load_weights(args.model)
    model_vars = [var for var in model.trainable_variables
                  if var.name.startswith("conv_mu") or var.name.startswith("group4/block_1/branch2b")]

    scod_model = SCOD(
        model=model,
        output_dist=GaussianDynamicDiagVar(),
        sketch_op_class=GaussianSketchOp,
        num_eigs=16,
        model_vars=model_vars,
        output_func=lambda x: x["mu"],
    )

    # build training dataset
    data_pipe = VODataPipe(p.data(batch_size=args.batch_size))
    train_ds = data_pipe.build_train_ds().repeat(args.repeat)
    if args.debug:
        train_ds = train_ds.take(10)

    scod_model.process_dataset(train_ds)
    # save SCOD model
    scod_model(model.input)
    scod_model.save(args.output)


import argparse

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

import model.detector
import utils.utils


if __name__ == '__main__':
    #指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/LPD.data',
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='weights/LPD-180-epoch-0.938281ap-model.pth',
                        help='The path of the .pth model to be transformed')
    parser.add_argument('--output', type=str, default='./model.onnx', 
                        help='The path where the onnx model is saved')

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)

    device = torch.device("cpu")

    # quantized_model = quantize_dynamic(opt.output, model_quant, weight_type=QuantType.QUInt8)

    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True, True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    #sets the module in eval node
    model.eval()

    test_data = torch.rand(1, 3, cfg["height"], cfg["width"]).to(device)
    torch.onnx.export(
        model,                      # model being run
        test_data,                  # model input (or a tuple for multiple inputs)
        opt.output,                 # where to save the model (can be a file or file-like object)
        export_params=True,         # store the trained parameter weights inside the model file
        opset_version=11,           # the ONNX version to export the model to
    #  weight_type=QuantType.QUInt8,
        do_constant_folding=True,   # whether to execute constant folding for optimization
    )  




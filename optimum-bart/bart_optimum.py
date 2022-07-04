# -*- coding: utf-8 -*-


from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import BartTokenizerFast
from pathlib import Path

checkpoint = "facebook/bart-large-cnn"
onnx_path = Path("onnx")

# Load vanilla transformers and convert to onnx
model = ORTModelForSeq2SeqLM.from_pretrained(checkpoint, from_transformers=True)
tokenizer = BartTokenizerFast.from_pretrained(checkpoint)

# save onnx checkpoint and tokenizer
model.save_pretrained(onnx_path)
tokenizer.save_pretrained(onnx_path)

# create ORTOptimizer and define optimization configuration
optimizer = ORTOptimizer.from_pretrained(checkpoint, feature=model.pipeline_task)
optimization_config = OptimizationConfig(optimization_level=99) # enable all optimizations

# apply the optimization configuration to the model
optimizer.export(
    onnx_model_path = onnx_path / "model.onnx",
    onnx_optimized_model_output_path = onnx_path / "model-optimized.onnx",
    optimization_config=optimization_config,
)

del model, optimizer

# Load optimized model
model = ORTModelForSeq2SeqLM.from_pretrained(onnx_path, file_name="model-optimized.onnx")
# create ORTQuantizer and define quantization configuration
dynamic_quantizer = ORTQuantizer.from_pretrained(checkpoint, feature=model.pipeline_task)
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
# apply the quantization configuration to the model
model_quantized_path = dynamic_quantizer.export(
    onnx_model_path=onnx_path / "model-optimized.onnx",
    onnx_quantized_model_output_path=onnx_path / "model-quantized.onnx",
    quantization_config=dqconfig,
)

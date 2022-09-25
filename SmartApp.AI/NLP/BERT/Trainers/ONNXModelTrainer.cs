using Microsoft.ML;
using System.Reflection;

namespace SmartBlazorApp.AI.NLP.BERT.Trainers;

public static class ONNXModelTrainer<T1>
    where T1 : class
{
    public static ITransformer BindAndTrains(
        MLContext _mlContext,
        string[] inputColumnNames,
        string[] outputColumnNames,
        string? modelPath = null,
        bool useGPU = false)
    {
        //need to swap to embedded resource for production environment eventually
        string bertModelPath = string.IsNullOrEmpty(modelPath) ?
            "ONNX/bertsquad-12.onnx"
            : modelPath;
        Microsoft.ML.Transforms.Onnx.OnnxScoringEstimator pipeline = _mlContext.Transforms.ApplyOnnxModel(
            modelFile: bertModelPath,
            inputColumnNames: inputColumnNames,
            outputColumnNames: outputColumnNames,
            gpuDeviceId: useGPU ? 0 : null);
        return pipeline.Fit(_mlContext.Data.LoadFromEnumerable(Enumerable.Empty<T1>()));
    }
}
using Microsoft.ML;
using SmartBlazorApp.AI.NLP.BERT.DataModel;

namespace SmartBlazorApp.AI.NLP.BERT.Predictors;

public class Predictor
{
    private readonly MLContext _mLContext;
    private PredictionEngine<BertInput, BertPredictions> _predictionEngine;

    public Predictor(ITransformer trainedModel)
    {
        _mLContext = new MLContext();
        _predictionEngine = _mLContext.Model.CreatePredictionEngine<BertInput, BertPredictions>(trainedModel);
    }

    public BertPredictions Predict(BertInput encodedInput) => _predictionEngine.Predict(encodedInput);
}

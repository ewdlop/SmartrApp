using SmartBlazorApp.AI.NLP.BERT.DataModel;
using SmartBlazorApp.AI.NLP.BERT.Predictors;
using SmartBlazorApp.AI.NLP.BERT.Tokenizers;
using SmartBlazorApp.AI.NLP.Extensions;
using SmartBlazorApp.AI.NLP.Helpers;

namespace SmartBlazorApp.AI.NLP.BERT;

/// <summary>
/// <see href="https://rubikscode.net/2021/04/19/machine-learning-with-ml-net-nlp-with-bert/"></see>
/// <para />
/// <see href="https://devblogs.microsoft.com/cesardelatorre/how-to-optimize-and-run-ml-net-models-on-scalable-asp-net-core-webapis-or-web-apps/#:~:text=Since%20the%20ML%20model%20%28ITransformer%29%20object%20is%20thread-safe%2C,Injection%20usage%2C%20as%20shown%20in%20the%20code%20below%3A"></see>
/// </summary>
public class BidirectionalEncoderRepresentationsFromTransformers
{
    private readonly List<string> _vocabulary;
    private readonly Tokenizer _tokenizer;
    private readonly Predictor<BertInput, BertPredictions> _predictor;

    public BidirectionalEncoderRepresentationsFromTransformers(List<string> vocabulary, string bertModelPath)
    {
        _vocabulary = vocabulary;
        _tokenizer = new Tokenizer(_vocabulary);
        _predictor = new Predictor<BertInput, BertPredictions>();
    }

    public (List<string> tokens, float probability) Predict(string context, string question)
    {
        List<(string Token, int VocabularyIndex, long SegmentIndex)> tokens = _tokenizer.Tokenize(question, context);
        BertInput input = BuildInput(tokens);

        BertPredictions predictions = _predictor.Predict(input);

        int contextStart = tokens.FindIndex(o => o.Token == Tokens.Separation);

        (int startIndex, int endIndex, float probability) = GetBestPrediction(predictions, contextStart, 20, 30);

        List<string> predictedTokens = input.InputIds
            .Skip(startIndex)
            .Take(endIndex + 1 - startIndex)
            .Select(o => _vocabulary[(int)o])
            .ToList();

        List<string> connectedTokens = Tokenizer.Untokenize(predictedTokens);

        return (connectedTokens, probability);
    }

    private static BertInput BuildInput(List<(string Token, int Index, long SegmentIndex)> tokens)
    {
        List<long> padding = Enumerable.Repeat(0L, 256 - tokens.Count).ToList();

        long[] tokenIndexes = tokens.Select(token => (long)token.Index).Concat(padding).ToArray();
        long[] segmentIndexes = tokens.Select(token => token.SegmentIndex).Concat(padding).ToArray();
        long[] inputMask = tokens.Select(o => 1L).Concat(padding).ToArray();

        return new BertInput()
        {
            InputIds = tokenIndexes,
            SegmentIds = segmentIndexes,
            InputMask = inputMask,
            UniqueIds = new long[] { 0 }
        };
    }

    private static (int StartIndex, int EndIndex, float Probability) GetBestPrediction(BertPredictions result, int minIndex, int topN, int maxLength)
    {
        IEnumerable<(float Logit, int Index)> bestStartLogits = result.StartLogits
            .Select((logit, index) => (Logit: logit, Index: index))
            .OrderByDescending(o => o.Logit)
            .Take(topN);

        IEnumerable<(float Logit, int Index)> bestEndLogits = result.EndLogits
            .Select((logit, index) => (Logit: logit, Index: index))
            .OrderByDescending(o => o.Logit)
            .Take(topN);

        IEnumerable<(int StartLogit, int EndLogit, float Score)> bestResultsWithScore = bestStartLogits
            .SelectMany(startLogit =>
                bestEndLogits
                .Select(endLogit =>
                    (
                        StartLogit: startLogit.Index,
                        EndLogit: endLogit.Index,
                        Score: startLogit.Logit + endLogit.Logit
                    )
                 )
            )
            .Where(entry => !(entry.EndLogit < entry.StartLogit
                              || entry.EndLogit - entry.StartLogit > maxLength
                              || entry.StartLogit == 0
                              && entry.EndLogit == 0
                              || entry.StartLogit < minIndex))
            .Take(topN);

        ((int StartLogit, int EndLogit, float Score) item, float probability) = bestResultsWithScore
            .Softmax(o => o.Score)
            .OrderByDescending(o => o.Probability)
            .FirstOrDefault();

        return (StartIndex: item.StartLogit, EndIndex: item.EndLogit, probability);
    }
}
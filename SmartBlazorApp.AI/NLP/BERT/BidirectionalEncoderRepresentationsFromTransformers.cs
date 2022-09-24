﻿using SmartBlazorApp.AI.NLP.BERT.DataModel;
using SmartBlazorApp.AI.NLP.BERT.Predictors;
using SmartBlazorApp.AI.NLP.BERT.Tokenizers;
using SmartBlazorApp.AI.NLP.BERT.Trainers;
using SmartBlazorApp.AI.NLP.Extensions;
using SmartBlazorApp.AI.NLP.Helpers;

namespace SmartBlazorApp.AI.NLP.BERT;

/// <summary>
/// https://rubikscode.net/2021/04/19/machine-learning-with-ml-net-nlp-with-bert/
/// </summary>
public class BidirectionalEncoderRepresentationsFromTransformers
{
    private readonly IList<ReadOnlyMemory<char>> _vocabulary;

    private readonly Tokenizer _tokenizer;
    private readonly Predictor _predictor;

    /// <summary>
    /// Use Onnx
    /// </summary>
    /// <param name="bertModelPath"></param>
    public BidirectionalEncoderRepresentationsFromTransformers(string bertModelPath, int size)
    {
        _vocabulary = new List<ReadOnlyMemory<char>>(size);
        _tokenizer = new Tokenizer(_vocabulary);
        Trainer trainer = new Trainer();
        Microsoft.ML.ITransformer trainedModel = trainer.BindAndTrains(bertModelPath, false);
        _predictor = new Predictor(trainedModel);
    }

    public async Task ReadVocabularyFilePAsync(string vocabularyFilePath)
    {
        await foreach (ReadOnlyMemory<char> vocab in vocabularyFilePath.ReadLinesAsMemoryAsync())
        {
            try
            {
                _vocabulary.Add(vocab);
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
            }
        }
    }

    public void ReadVocabularyFile(string vocabularyFilePath)
    {
        foreach (string? vocab in vocabularyFilePath.ReadLines())
        {
            try
            {
                _vocabulary.Add(vocab.AsMemory());
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
            }
        }
    }

    public (List<string> tokens, float probability) Predict(string context, string question)
    {
        List<(ReadOnlyMemory<char> Token, int VocabularyIndex, long SegmentIndex)> tokens = _tokenizer.Tokenize(question, context);
        BertInput input = BuildInput(tokens);
        BertPredictions predictions = _predictor.Predict(input);
        int contextStart = tokens.FindIndex(o => o.Token.Span.SequenceEqual(Tokens.Separation.Span));
        (int startIndex, int endIndex, float probability) = GetBestPrediction(predictions, contextStart, 20, 30);


        List<ReadOnlyMemory<char>>? predictedTokens = input.InputIds?
         .Skip(startIndex)
         .Take(endIndex + 1 - startIndex)
         .Select(o => _vocabulary[(int)o])
         .ToList() ?? new List<ReadOnlyMemory<char>>();

        List<string> connectedTokens = predictedTokens.ToUntokenizedString();

        return (connectedTokens, probability);
    }

    private static BertInput BuildInput(List<(ReadOnlyMemory<char> Token, int Index, long SegmentIndex)> tokens)
    {
        IEnumerable<long> padding = Enumerable.Repeat(0L, 256 - tokens.Count);
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
        IEnumerable<(float Logit, int Index)> bestStartLogits = result.StartLogistics?
            .Select((logit, index) => (Logit: logit, Index: index))
            .OrderByDescending(o => o.Logit)
            .Take(topN) ?? Enumerable.Empty<(float Logit, int Index)>();

        IEnumerable<(float Logit, int Index)> bestEndLogits = result.EndLogistics?
            .Select((logit, index) => (Logit: logit, Index: index))
            .OrderByDescending(o => o.Logit)
            .Take(topN) ?? Enumerable.Empty<(float Logit, int Index)>();

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
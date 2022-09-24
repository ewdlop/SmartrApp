using Microsoft.ML.Data;

namespace SmartBlazorApp.AI.NLP.BERT.DataModel;

public class BertInput
{
    [VectorType(1)]
    [ColumnName("unique_ids_raw_output___9:0")]
    public long[]? UniqueIds { get; init; }

    [VectorType(1, 256)]
    [ColumnName("segment_ids:0")]
    public long[]? SegmentIds { get; init; }

    [VectorType(1, 256)]
    [ColumnName("input_mask:0")]
    public long[]? InputMask { get; init; }

    [VectorType(1, 256)]
    [ColumnName("input_ids:0")]
    public long[]? InputIds { get; init; }
}
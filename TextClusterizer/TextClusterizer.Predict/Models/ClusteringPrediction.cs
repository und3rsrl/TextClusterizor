using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace TextClusterizer.Predict.Models
{
    public class ClusteringPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint SelectedClusterId;
        [ColumnName("Score")]
        public float[] Distance;
        [ColumnName("PCAFeatures")]
        public float[] Location;
        [ColumnName("ArticleText")]
        public string Article;
    }
}

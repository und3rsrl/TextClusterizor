using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace TextClusterizer.Model.Models
{
    /// <summary>
    /// A prediction class that holds a single cluster prediction.
    /// </summary>
    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId { get; set; }

        [ColumnName("Score")]
        public float[] Distances { get; set; }
    }
}

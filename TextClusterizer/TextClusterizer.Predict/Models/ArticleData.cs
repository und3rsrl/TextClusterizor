using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace TextClusterizer.Model.Models
{
    /// <summary>
    /// A data transfer class that holds a single article.
    /// </summary>
    public class ArticleData
    {
        [LoadColumn(0)]
        public string ArticleText { get; set; }
    }
}

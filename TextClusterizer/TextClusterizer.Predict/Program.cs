using Microsoft.ML;
using System;
using System.IO;

namespace TextClusterizer.Predict
{
    class Program
    {
        static void Main(string[] args)
        {
            var assetsRelativePath = @"../../../Assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);
            var articlesCsv = Path.Combine(assetsPath, "Input", "articles.csv");
            var modelPath = Path.Combine(assetsPath, "Input", "articlesClustering.zip");
            var plotSvg = Path.Combine(assetsPath, "Output", "articles.svg");
            var plotCsv = Path.Combine(assetsPath, "Output", "articles.csv");

            try
            {
                MLContext mlContext = new MLContext();
                var clusteringModelScorer = new ClusteringModelScorer(mlContext, articlesCsv, plotSvg, plotCsv);
                clusteringModelScorer.LoadModel(modelPath);

                clusteringModelScorer.CreateArticlesClusters();
            }
            catch (Exception ex)
            {
                Helpers.ConsoleHelper.ConsoleWriteException(ex.ToString());
            }

            Helpers.ConsoleHelper.ConsolePressAnyKey();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}

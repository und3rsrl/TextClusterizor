using Microsoft.ML;
using System;
using System.IO;
using TextClusterizer.Model.Helpers;
using TextClusterizer.Model.Models;

namespace TextClusterizer.Model
{
    class Program
    {
        static void Main(string[] args)
        {
            string assetsRelativePath = @"../../../Assets";
            string assetsPath = PathHelper.GetAbsolutePath(assetsRelativePath);

            string articlesCsv = Path.Combine(assetsPath, "Input", "articles.csv");
            string modelPath = Path.Combine(assetsPath, "Output", "articlesClustering.zip");

            var mlContext = new MLContext();

            var trainingData = mlContext.Data.LoadFromTextFile<ArticleData>(articlesCsv);

            var data = trainingData.Preview();

            var dataProcessPipeline = mlContext.Transforms.Text.NormalizeText(outputColumnName: Constants.Columns.NormalizedTextColumn, inputColumnName: Constants.Columns.ArticleTextColumn)
                .Append(mlContext.Transforms.Text
                    .TokenizeIntoWords
                    (
                        outputColumnName: Constants.Columns.TokenizedTextColumn,
                        inputColumnName: Constants.Columns.NormalizedTextColumn
                    ))
                .Append(mlContext.Transforms.Text
                    .RemoveDefaultStopWords
                    (
                        outputColumnName: Constants.Columns.WordsWithoutStopWordsENColumn,
                        inputColumnName: Constants.Columns.TokenizedTextColumn,
                        language: Microsoft.ML.Transforms.Text.StopWordsRemovingEstimator.Language.English
                    ))
                .Append(mlContext.Transforms.Text
                    .RemoveDefaultStopWords
                    (
                        outputColumnName: Constants.Columns.WordsWithoutStopWordsFRColumn,
                        inputColumnName: Constants.Columns.WordsWithoutStopWordsENColumn,
                        language: Microsoft.ML.Transforms.Text.StopWordsRemovingEstimator.Language.French
                    ))
                .Append(mlContext.Transforms.Text
                    .RemoveDefaultStopWords
                    (
                        outputColumnName: Constants.Columns.WordsWithoutStopWordsDEColumn,
                        inputColumnName: Constants.Columns.WordsWithoutStopWordsFRColumn,
                        language: Microsoft.ML.Transforms.Text.StopWordsRemovingEstimator.Language.German
                    ))
                .Append(mlContext.Transforms.Conversion
                    .MapValueToKey
                    (
                        outputColumnName: Constants.Columns.ValueToKeyColumn,
                        inputColumnName: Constants.Columns.WordsWithoutStopWordsDEColumn
                    ))
                .Append(mlContext.Transforms.Text
                    .ProduceNgrams
                    (
                        outputColumnName: Constants.Columns.NGramsColumn,
                        inputColumnName: Constants.Columns.ValueToKeyColumn
                    ))
                .Append(mlContext.Transforms
                    .NormalizeLpNorm
                    (
                        outputColumnName: Constants.Columns.NormalizeLpNormColumn,
                        inputColumnName: Constants.Columns.NGramsColumn
                    ))
                .Append(mlContext.Transforms
                    .Concatenate
                    (
                        Constants.Columns.FeaturesColumn,
                        Constants.Columns.NormalizeLpNormColumn
                    ));

            var transformedData = dataProcessPipeline.Fit(trainingData).Transform(trainingData).Preview();

            var trainer = mlContext.Clustering.Trainers.KMeans(featureColumnName: "Features", numberOfClusters: 3);
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            Console.WriteLine("Start training model....");
            ITransformer trainedModel = trainingPipeline.Fit(trainingData);
            Console.WriteLine("Model training complete!");

            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            var predictions = trainedModel.Transform(trainingData);
            var metrics = mlContext.Clustering.Evaluate(predictions, scoreColumnName: "Score", featureColumnName: "Features");

            ConsoleHelper.PrintClusteringMetrics(trainer.ToString(), metrics);
            mlContext.Model.Save(trainedModel, trainingData.Schema, modelPath);
            Console.WriteLine("The model is saved to {0}", modelPath);
        }
    }
}

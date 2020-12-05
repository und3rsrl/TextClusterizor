using Microsoft.ML;
using OxyPlot;
using OxyPlot.Series;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using TextClusterizer.Model.Models;
using TextClusterizer.Predict.Models;

namespace TextClusterizer.Predict
{
    public class ClusteringModelScorer
    {
        private readonly string _articlesDataLocation;
        private readonly string _plotLocation;
        private readonly string _csvlocation;
        private readonly MLContext _mlContext;
        private ITransformer _trainedModel;

        public ClusteringModelScorer(MLContext mlContext, string pivotDataLocation, string plotLocation, string csvlocation)
        {
            _articlesDataLocation = pivotDataLocation;
            _plotLocation = plotLocation;
            _csvlocation = csvlocation;
            _mlContext = mlContext;
        }

        public ITransformer LoadModel(string modelPath)
        {
            _trainedModel = _mlContext.Model.Load(modelPath, out var modelInputSchema);
            return _trainedModel;
        }

        public void CreateArticlesClusters()
        {
            var data = _mlContext.Data.LoadFromTextFile<ArticleData>(_articlesDataLocation, hasHeader: true);

            //Apply data transformation to create predictions/clustering
            var tranfomedDataView = _trainedModel.Transform(data);
            var predictions = _mlContext.Data.CreateEnumerable<ClusteringPrediction>(tranfomedDataView, false)
                            .ToArray();

            //Generate data files with customer data grouped by clusters
            SaveArticlesCSV(predictions, _csvlocation);

            //Plot/paint the clusters in a chart and open it with the by-default image-tool in Windows
            SaveArticlesPlotChart(predictions, _plotLocation);
            OpenChartInDefaultWindow(_plotLocation);
        }

        private static void SaveArticlesCSV(IEnumerable<ClusteringPrediction> predictions, string csvlocation)
        {
            Helpers.ConsoleHelper.ConsoleWriteHeader("CSV Articles");
            using (var w = new System.IO.StreamWriter(csvlocation))
            {
                w.WriteLine($"Article,SelectedClusterId");
                w.Flush();
                predictions.ToList().ForEach(prediction => {
                    w.WriteLine($"{prediction.Article},{prediction.SelectedClusterId}");
                    w.Flush();
                });
            }

            Console.WriteLine($"CSV location: {csvlocation}");
        }

        private static void SaveArticlesPlotChart(IEnumerable<ClusteringPrediction> predictions, string plotLocation)
        {
            Helpers.ConsoleHelper.ConsoleWriteHeader("Plot Articles");

            var plot = new PlotModel { Title = "Articles", IsLegendVisible = true };

            var clusters = predictions.Select(p => p.SelectedClusterId).Distinct().OrderBy(x => x);

            foreach (var cluster in clusters)
            {
                var scatter = new ScatterSeries { MarkerType = MarkerType.Circle, MarkerStrokeThickness = 2, Title = $"Cluster: {cluster}", RenderInLegend = true };
                var series = predictions
                    .Where(p => p.SelectedClusterId == cluster)
                    .Select(p => new ScatterPoint(p.Location[0], p.Location[1])).ToArray();
                scatter.Points.AddRange(series);
                plot.Series.Add(scatter);
            }

            plot.DefaultColors = OxyPalettes.HueDistinct(plot.Series.Count).Colors;

            var exporter = new SvgExporter { Width = 600, Height = 400 };
            using (var fs = new System.IO.FileStream(plotLocation, System.IO.FileMode.Create))
            {
                exporter.Export(plot, fs);
            }

            Console.WriteLine($"Plot location: {plotLocation}");
        }

        private static void OpenChartInDefaultWindow(string plotLocation)
        {
            Console.WriteLine("Showing chart...");
            var p = new Process();
            p.StartInfo = new ProcessStartInfo(plotLocation)
            {
                UseShellExecute = true
            };
            p.Start();
        }
    }
}

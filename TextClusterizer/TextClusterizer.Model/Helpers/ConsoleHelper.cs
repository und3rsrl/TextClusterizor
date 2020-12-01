using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace TextClusterizer.Model.Helpers
{
    public class ConsoleHelper
    {
        public static void PrintClusteringMetrics(string name, ClusteringMetrics metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for {name} clustering model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       Normalized Mutual Information is: {metrics.NormalizedMutualInformation}");
            Console.WriteLine($"*       Average Distance: {metrics.AverageDistance}");
            Console.WriteLine($"*       Davies Bouldin Index is: {metrics.DaviesBouldinIndex}");
            Console.WriteLine($"*************************************************");
        }
    }
}

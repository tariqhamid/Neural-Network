using System;
using NN;
using System.IO;

namespace TestApp
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork nn;
            Console.WriteLine("Please chose a method for learning: ");
            Console.WriteLine("0: Back Propagation (Fast)");
            Console.WriteLine("1: Genetic Algorithm (Slow)");
            Console.WriteLine("2: Particle Swarm Optimisation (Recomended)");
            while (true)
            {
                string str = Console.ReadLine();
                if (str.Length > 0)
                {
                    if (str[0] == '0')
                    {
                        nn = BP();
                        break;
                    }
                    else if (str[0] == '1')
                    {
                        nn = GA();
                        break;
                    }
                    else if (str[0] == '2')
                    {
                        nn = PSO();
                        break;
                    }
                    else
                    {
                        Console.WriteLine("Please enter 0, 1 or 2 to chose which type of learning method you would like to use");
                    }
                }
                else
                    Console.WriteLine("Please enter 0, 1 or 2 to chose which type of learning method you would like to use");
            }

            Console.WriteLine("You can now enter inputs and the neural network will try to solve them\ne.g. \"5.9,3.0,5.1,1.8\"");

            while (true)
            {
                string cmd = Console.ReadLine();
                if (cmd.Length > 7)
                {
                    string[] str = cmd.Split(',');
                    double[] inputs = new double[str.Length];

                    for (int i = 0; i < inputs.Length; i++)
                        inputs[i] = double.Parse(str[i]);

                    double[] output = nn.ComputeResults(inputs);

                    for (int i = 0; i < output.Length; i++)
                        Console.Write(output[i] + ", ");

                    Console.WriteLine();
                }
            }
        }

        private static NeuralNetwork PSO()
        {
            ParticleSwarmOptimisation pso = new ParticleSwarmOptimisation(4, 7, 3, ActivationType.HyperbolicTangent, ActivationType.Softmax);

            Console.WriteLine("Loading Data");
            double[][] data = GetData();
            Console.WriteLine("Data contains " + data.Length + " results");
            Console.WriteLine("Starting Training");

            pso.Train(data,//Data
                12,//Amount of particles
                0.01,//Exit when error is as low as this value
                0.005,//death probability
                1000//repeat
                );

            Console.WriteLine("Network Trained");
            Console.WriteLine("Acuracy on train data: " + pso.Accuracy(data));
            Console.WriteLine("Final Weights (Rounded): ");
            double[] weights = pso.GetWeights();
            for (int i = 0; i < weights.Length; i++)
            {
                if (weights[i] > 0)
                    Console.Write("+{0:00.000} ", weights[i]);
                else
                    Console.Write("{0:00.000} ", weights[i]);
            }
            Console.WriteLine();
            Console.WriteLine("Done!");

            return pso;
        }

        private static NeuralNetwork BP()
        {
            BackPropagation bp = new BackPropagation(4, 7, 3, ActivationType.HyperbolicTangent, ActivationType.Softmax);

            Console.WriteLine("Loading Data");

            double[][] data = GetData();

            Console.WriteLine("Data contains " + data.Length + " results");

            Console.WriteLine("Starting Training");

            bp.Train(data //Data
                , 0.05 //Learning rate
                , 0.01 //momentum
                , 0.0001 //decay
                , 5000 //repeat
                );

            Console.WriteLine("Network Trained");
            Console.WriteLine("Acuracy on train data: " + bp.Accuracy(data));
            Console.WriteLine("Final Weights (Rounded): ");
            double[] weights = bp.GetWeights();
            for (int i = 0; i < weights.Length; i++)
            {
                if (weights[i] > 0)
                    Console.Write("+{0:00.000} ", weights[i]);
                else
                    Console.Write("{0:00.000} ", weights[i]);
            }
            Console.WriteLine();
            Console.WriteLine("Done!");

            return bp;
        }

        private static NeuralNetwork GA()
        {
            GeneticAlgorithm ga = new GeneticAlgorithm(4, 7, 3, ActivationType.HyperbolicTangent, ActivationType.Softmax);

            Console.WriteLine("Loading Data");

            double[][] data = GetData();

            Console.WriteLine("Data contains " + data.Length + " results");

            Console.WriteLine("Starting Training");

            ga.Train(data, //Data
                5000, //Generations
                20, //Population Size
                5); //Mutation

            Console.WriteLine("Network Trained");
            Console.WriteLine("Acuracy on train data: " + ga.Accuracy(data));
            Console.WriteLine("Final Weights (Rounded): ");
            double[] weights = ga.GetWeights();
            for (int i = 0; i < weights.Length; i++)
            {
                if (weights[i] > 0)
                    Console.Write("+{0:00.000} ", weights[i]);
                else
                    Console.Write("{0:00.000} ", weights[i]);
            }
            Console.WriteLine();
            Console.WriteLine("Done!");

            return ga;
        }

        //Reads the data from text file
        private static double[][] GetData()
        {
            TextReader read = new StreamReader("../../../data.txt");
            string s = read.ReadToEnd();
            read.Close();

            string[] data = s.Split(';');
            double[][] result = new double[data.Length][];

            for (int i = 0; i < data.Length; i++)
            {
                string[] io = data[i].Split(',');
                result[i] = new double[io.Length];
                for (int j = 0; j < io.Length; j++)
                {
                    result[i][j] = double.Parse(io[j]);
                }
            }
            return result;
        }
    }
}
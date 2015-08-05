using System;
using System.Collections.Generic;
using System.Text;

namespace NN
{
    /// <summary>
    /// Uses Genetic Algorithm To Train Neural Network
    /// Genetic algorithm is genrally really slow but they should give the best result.
    /// </summary>
    public class GeneticAlgorithm : NeuralNetwork
    {
        #region Constructors

        /// <summary>
        /// Initialise Neural Network and Genetic Algorithm learning
        /// </summary>
        /// <param name="input">Length of input layer</param>
        /// <param name="hidden">Length of hidden layer</param>
        /// <param name="output">Length of output layer</param>
        /// <param name="hType">which type of activation method would be used in hidden</param>
        /// <param name="oType">which type of activation method would be used in output</param>
        public GeneticAlgorithm(int input, int hidden, int output, ActivationType hType, ActivationType oType)
            : base(input, hidden, output, hType, oType)
        {

        }

        /// <summary>
        /// Initialise Neural Network and Genetic Algorithm learning
        /// </summary>
        /// <param name="nn">Use Existing Neural Network</param>
        public GeneticAlgorithm(NeuralNetwork nn)
            : base(nn.Input, nn.Hidden, nn.Output, nn.Hidden_Type, nn.Output_Type)
        {

        } 

        #endregion

        public void Train(double[][] data, int generations, int amount, double mutation)
        {
            if (amount < 2)
                throw new Exception("Amount must be bigger or equal to 2");

            int g = 0;

            Random rnd = new Random();
            NeuralNetwork[] nn = new NeuralNetwork[amount];
            int[] order = new int[amount];
            double[] acuracy = new double[amount];
            double BestAcuracy = 0;

            //Initial
            for (int i = 0; i < amount; i++)
            {
                nn[i] = new NeuralNetwork(this.Input, this.Hidden, this.Output, this.Hidden_Type, this.Output_Type);
                nn[i].SetWeights(this.GetWeights());
            }

            while (g < generations)
            {
                //Calculate acuracy
                for (int i = 0; i < amount; i++)
                {
                    acuracy[i] = nn[i].Accuracy(data);
                }

                //Get Best results
                if (BestAcuracy < acuracy[order[0]])
                {
                    BestAcuracy = acuracy[order[0]];
                    this.SetWeights(nn[order[0]].GetWeights());
                }

                //Evolve/Selection
                for (int i = 0; i < amount; i++)
                {
                    for (int j = i; j < amount; j++)
                    {
                        if (acuracy[j] > acuracy[i])
                        {
                            order[j] = i;
                            order[i] = j;
                        }
                    }
                }

                //Mutate
                for (int i = 0; i < amount; i++)
                {
                    double[] weights = nn[order[i]].GetWeights();
                    for (int j = 0; j < weights.Length; j++)
                    {
                        double multiplier = acuracy[order[i]]; // Makes sure as acuracy gets closer to 1 the results change less
                        double m = (((rnd.NextDouble() - 0.5) * 2) / multiplier);

                        weights[j] = m * mutation * (i + 1);
                    }

                    nn[order[i]].SetWeights(weights);
                }

                
                //Cross-Over
                for (int i = 0; i < (amount / 2); i++)
                {
                    double[] w1 = nn[order[(i * 2) + 0]].GetWeights();
                    double[] w2 = nn[order[(i * 2) + 0]].GetWeights();
                    double[] j1 = new double[w1.Length];
                    double[] j2 = new double[w1.Length];
                    for (int j = 0; j < w1.Length; j++)
                    {
                        if (j % 2 == 0)
                        {
                            j1[j] = w1[j];
                            j2[j] = w2[j];
                        }
                        else
                        {
                            j1[j] = w2[j];
                            j2[j] = w1[j];
                        }
                    }

                    nn[order[(i * 2) + 0]].SetWeights(j1);
                    nn[order[(i * 2) + 0]].SetWeights(j2);
                }

                Console.Write("\rLearning: " + (g + 1) + "/" + generations + " Acuracy: " + BestAcuracy);
                g++;
            }
            Console.WriteLine();
        }
    }
}
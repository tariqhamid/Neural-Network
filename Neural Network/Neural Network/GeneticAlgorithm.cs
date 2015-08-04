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
    }
}
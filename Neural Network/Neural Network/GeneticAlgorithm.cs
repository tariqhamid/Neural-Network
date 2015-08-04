using System;
using System.Collections.Generic;
using System.Text;

namespace NN
{
    public class GeneticAlgorithm : NeuralNetwork
    {

        #region Constructors

        public GeneticAlgorithm(int input, int hidden, int output, ActivationType hType, ActivationType oType)
            : base(input, hidden, output, hType, oType)
        {

        }

        public GeneticAlgorithm(NeuralNetwork nn)
            : base(nn.Input, nn.Hidden, nn.Output, nn.Hidden_Type, nn.Output_Type)
        {

        } 

        #endregion
    }
}
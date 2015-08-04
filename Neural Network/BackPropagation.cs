using System;
using System.Collections.Generic;
using System.Text;

namespace NN
{
    /// <summary>
    /// Uses Back Propagation To Train Neural Network
    /// Back Propagation is the fastest technique but is not very accurate
    /// </summary>
    public class BackPropagation : NeuralNetwork
    {
        #region Variables

        private double[] oGrads;
        private double[] hGrads;

        private double[][] ihPrevWeightsDelta;
        private double[][] hoPrevWeightsDelta;

        private double[] hPrevBiasesDelta;
        private double[] oPrevBiasesDelta;

        #endregion

        #region Constructor

        /// <summary>
        /// Initialise Neural Network and Back Propagation learning
        /// </summary>
        /// <param name="input">Length of input layer</param>
        /// <param name="hidden">Length of hidden layer</param>
        /// <param name="output">Length of output layer</param>
        /// <param name="hType">which type of activation method would be used in hidden</param>
        /// <param name="oType">which type of activation method would be used in output</param>
        public BackPropagation(int input, int hidden, int output, ActivationType hType, ActivationType oType)
            : base(input, hidden, output, hType, oType)
        {
            if (oType == ActivationType.HeavisideStep || hType == ActivationType.HeavisideStep)
                throw new Exception("Back Propagation does not support heavisidestep");

            oGrads = new double[output];
            hGrads = new double[hidden];

            hPrevBiasesDelta = new double[hidden];
            oPrevBiasesDelta = new double[output];

            ihPrevWeightsDelta = new double[input][];
            for (int i = 0; i < input; i++)
                ihPrevWeightsDelta[i] = new double[hidden];

            hoPrevWeightsDelta = new double[hidden][];
            for (int i = 0; i < hidden; i++)
                hoPrevWeightsDelta[i] = new double[output];
        }

        /// <summary>
        /// Initialise Neural Network and Back Propagation learning
        /// </summary>
        /// <param name="nn">Use Existing Neural Network</param>
        public BackPropagation(NeuralNetwork nn)
            : base(nn.Input, nn.Hidden, nn.Output, nn.Hidden_Type, nn.Output_Type)
        {
            if (nn.Output_Type == ActivationType.HeavisideStep || nn.Hidden_Type == ActivationType.HeavisideStep)
                throw new Exception("Back Propagation does not support heavisidestep");

            oGrads = new double[nn.Output];
            hGrads = new double[nn.Hidden];

            hPrevBiasesDelta = new double[nn.Hidden];
            oPrevBiasesDelta = new double[nn.Output];

            ihPrevWeightsDelta = new double[nn.Input][];
            for (int i = 0; i < nn.Input; i++)
                ihPrevWeightsDelta[i] = new double[nn.Hidden];

            hoPrevWeightsDelta = new double[nn.Hidden][];
            for (int i = 0; i < nn.Hidden; i++)
                hoPrevWeightsDelta[i] = new double[nn.Output];
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Use this method to train the neural network
        /// </summary>
        /// <param name="data">A array containning the input and output for the neural network to learn</param>
        /// <param name="learnRate">How fast network should learn</param>
        /// <param name="momentum">How fast should neural network change</param>
        /// <param name="weightDecay">How fast should neural network forget</param>
        /// <param name="repeat">How many times would you like to repeat the trainning data</param>
        public void Train(double[][] data, double learnRate, double momentum, double weightDecay, int repeat)
        {
            if (data[0].Length != this.Input + this.Output)
                throw new Exception("Array input and outputs don't match");
            int r = 0;

            while (r < repeat)
            {
                int[] shuffled = new int[data.Length];
                for (int i = 0; i < shuffled.Length; i++)
                    shuffled[i] = i;
                shuffled = Shuffle(shuffled);

                for (int i = 0; i < data.Length; i++)
                {
                    int k = 0;
                    double[] inputs = new double[this.Input];
                    double[] targets = new double[this.Output];

                    for (int j = 0; j < this.Input; j++)
                        inputs[j] = data[shuffled[i]][k++];
                    for (int j = 0; j < this.Output; j++)
                        targets[j] = data[shuffled[i]][k++];

                    double[] result = UpdateNeuralNetwork(inputs, targets, learnRate, momentum, weightDecay);
                    if (result[0] == double.NaN)
                    {
                        throw new Exception("Doube is NaN");
                    }
                }
                Console.Write("\rLearning: " + (r + 1) + "/" + repeat);
                r++;
            }

            Console.WriteLine();
        }

        #endregion

        #region Private Methods

        private double[] UpdateNeuralNetwork(double[] input, double[] target, double learnRate, double momentum, double weightDecay)
        {
            if (input.Length != this.Input)
                throw new Exception("Input length doesn't match");
            if (target.Length != this.Output)
                throw new Exception("target length doesn't match");

            double[] output = ComputeResults(input);

            //Output gradients
            for (int i = 0; i < oGrads.Length; i++)
            {
                double derivative = output[i];
                switch (Output_Type)
                {
                    case ActivationType.Softmax:
                    case ActivationType.LogisticSigmoid:
                        derivative = ActivationMethods.LogisticSigmoidDerivative(output[i]);
                        break;
                    case ActivationType.HyperbolicTangent:
                        derivative = ActivationMethods.HyperbolicTangentDerivative(output[i]);
                        break;
                }
                oGrads[i] = derivative * (target[i] - output[i]);
            }

            //Hidden gradiants
            for (int i = 0; i < hGrads.Length; i++)
            {
                double derivative = hOutputs[i];
                switch (Hidden_Type)
                {
                    case ActivationType.Softmax:
                    case ActivationType.LogisticSigmoid:
                        derivative = ActivationMethods.LogisticSigmoidDerivative(hOutputs[i]);
                        break;
                    case ActivationType.HyperbolicTangent:
                        derivative = ActivationMethods.HyperbolicTangentDerivative(hOutputs[i]);
                        break;
                }

                double sum = 0.0;
                for (int j = 0; j < Output; ++j) // each hidden delta is the sum of numOutput terms
                {
                    double x = oGrads[j] * HO_Weights[i][j];
                    sum += x;
                }
                hGrads[i] = derivative * sum;
            }

            //Update hidden Weights
            for (int i = 0; i < Input; i++)
            {
                for (int j = 0; j < Hidden; j++)
                {
                    double delta = learnRate * hGrads[j] * input[i]; // compute the new delta
                    IH_Weights[i][j] += delta; // update. note we use '+' instead of '-'. this can be very tricky.
                    // now add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
                    IH_Weights[i][j] += momentum * ihPrevWeightsDelta[i][j];
                    IH_Weights[i][j] -= (weightDecay * IH_Weights[i][j]); // weight decay
                    ihPrevWeightsDelta[i][j] = delta; // don't forget to save the delta for momentum 

                }
            }

            //Update hidden biases
            for (int i = 0; i < Hidden; ++i)
            {
                double delta = learnRate * hGrads[i] * 1.0; // t1.0 is constant input for bias; could leave out
                Hidden_Biases[i] += delta;
                Hidden_Biases[i] += momentum * hPrevBiasesDelta[i]; // momentum
                Hidden_Biases[i] -= (weightDecay * Hidden_Biases[i]); // weight decay
                hPrevBiasesDelta[i] = delta; // don't forget to save the delta
            }


            //Update output weights
            for (int i = 0; i < Hidden; ++i)
            {
                for (int j = 0; j < Output; ++j)
                {
                    // see above: hOutputs are inputs to the nn outputs
                    double delta = learnRate * oGrads[j] * hOutputs[i];
                    HO_Weights[i][j] += delta;
                    HO_Weights[i][j] += momentum * hoPrevWeightsDelta[i][j]; // momentum
                    HO_Weights[i][j] -= (weightDecay * HO_Weights[i][j]); // weight decay
                    hoPrevWeightsDelta[i][j] = delta; // save
                }
            }

            //Update output biases
            for (int i = 0; i < Output; ++i)
            {
                double delta = learnRate * oGrads[i] * 1.0;
                Output_Biases[i] += delta;
                Output_Biases[i] += momentum * oPrevBiasesDelta[i]; // momentum
                Output_Biases[i] -= (weightDecay * Output_Biases[i]); // weight decay
                oPrevBiasesDelta[i] = delta; // save
            }

            return oGrads;
        }

        #endregion
    }
}
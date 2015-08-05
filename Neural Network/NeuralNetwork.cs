using System;
using System.Collections.Generic;
using System.Text;

namespace NN
{
    /// <summary>
    /// Please use the following classes instead for creating and training the neural network
    /// BackPropagation
    /// ParticleSwarmOptimisation
    /// GeneticAlgorithm
    /// </summary>
    public class NeuralNetwork
    {
        #region Variables

        /// <summary>
        /// How many nodes are in input layer
        /// </summary>
        public int Input
        {
            get
            {
                return input;
            }
            private set
            {
                input = value;
            }
        }
        private int input;

        /// <summary>
        /// How many nodes are in hidden layer
        /// </summary>
        public int Hidden
        {
            get
            {
                return hidden;
            }
            private set
            {
                hidden = value;
            }
        }
        private int hidden;

        /// <summary>
        /// how many nodes are in output layer
        /// </summary>
        public int Output
        {
            get
            {
                return output;
            }
            private set
            {
                output = value;
            }
        }
        private int output;

        /// <summary>
        /// Which type of activation should be used for hidden layer
        /// </summary>
        public ActivationType Hidden_Type
        {
            get
            {
                return hType;
            }
            private set
            {
                hType = value;
            }
        }
        private ActivationType hType;
        /// <summary>
        /// Which type of activation should be used for output layer
        /// </summary>
        public ActivationType Output_Type
        {
            get
            {
                return oType;
            }
            private set
            {
                oType = value;
            }
        }
        private ActivationType oType;

        /// <summary>
        /// Contains values for hidden layers (Use this to create better deep learning algorithm)
        /// </summary>
        protected double[] hOutputs;

        //////////////////////////////////

        /// <summary>
        /// Array of weights linking input and hidden layers
        /// </summary>
        public double[][] IH_Weights
        {
            get
            {
                return ihweights;
            }
            protected set
            {
                howeights = value;
            }
        }
        private double[][] ihweights;
        /// <summary>
        /// Array of weights linking hidden and output layers
        /// </summary>
        public double[][] HO_Weights
        {
            get
            {
                return howeights;
            }
            protected set
            {
                howeights = value;
            }
        }
        private double[][] howeights;

        //////////////////////////////////

        /// <summary>
        /// biases for hidden layer
        /// </summary>
        public double[] Hidden_Biases
        {
            get
            {
                return hiddenBiases;
            }
            protected set
            {
                hiddenBiases = value;
            }
        }
        private double[] hiddenBiases;

        /// <summary>
        /// biases for output layer
        /// </summary>
        public double[] Output_Biases
        {
            get
            {
                return outputBiases;
            }
            protected set
            {
                outputBiases = value;
            }
        }
        private double[] outputBiases;

        public int WeightsLength
        {
            get
            {
                return (this.input * this.hidden) + (this.hidden * this.output) + this.hidden + this.output;
            }
        }

        #endregion

        #region Constructor

        /// <summary>
        /// Initialise Neural Network
        /// </summary>
        /// <param name="input">Length of input layer</param>
        /// <param name="hidden">Length of hidden layer</param>
        /// <param name="output">Length of output layer</param>
        /// <param name="hType">which type of activation method would be used in hidden</param>
        /// <param name="oType">which type of activation method would be used in output</param>
        public NeuralNetwork(int input, int hidden, int output, ActivationType hType, ActivationType oType)
        {
            //Setup Lengths
            this.input = input;
            this.hidden = hidden;
            this.output = output;
            this.hType = hType;
            this.oType = oType;

            //Setup Biases
            hiddenBiases = new double[hidden];
            outputBiases = new double[output];

            //Setup Weights
            ihweights = new double[input][];
            for (int i = 0; i < input; i++)
                ihweights[i] = new double[hidden];
            
            howeights = new double[hidden][];
            for (int  i = 0; i < hidden; i++)
                howeights[i] = new double[output];

            hOutputs = new double[hidden];

            GenerateWeights();
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Use Neural Network to calculate the output
        /// </summary>
        /// <param name="input">array of inputs for calculating output NOTE: array length must equal input</param>
        /// <returns></returns>
        public double[] ComputeResults(double[] input)
        {
            if (input.Length != this.input)
                throw new Exception("Input length doesn't match");
            
            double[] hidden = new double[this.hidden];
            double[] output = new double[this.output];

            #region Input -> Hidden
            //Weights
            for (int i = 0; i < this.input; i++)
            {
                for (int j = 0; j < this.hidden; j++)
                {
                    hidden[j] += ihweights[i][j] * input[i];
                }
            }

            //Biases
            for (int i = 0; i < this.hidden; i++)
            {
                hidden[i] += hiddenBiases[i];
            }

            //Activation
            switch(hType)
            {
                case ActivationType.HeavisideStep:
                    for (int i = 0; i < this.hidden; i++)
                        hOutputs[i] = ActivationMethods.HeavisideStep(hidden[i]);
                    break;
                case ActivationType.HyperbolicTangent:
                    for (int i = 0; i < this.hidden; i++)
                        hOutputs[i] = ActivationMethods.HyperbolicTangent(hidden[i]);
                    break;
                case ActivationType.LogisticSigmoid:
                    for (int i = 0; i < this.hidden; i++)
                        hOutputs[i] = ActivationMethods.LogisticSigmoid(hidden[i]);
                    break;
                case ActivationType.Softmax:
                    hOutputs = ActivationMethods.Softmax(hidden);
                    break;
            }

            #endregion

            #region Hidden -> Output
            //Weights
            for (int i = 0; i < this.hidden; i++)
            {
                for (int j = 0; j < this.output; j++)
                {
                    output[j] += howeights[i][j] * hOutputs[i];
                }
            }

            //Biases
            for (int i = 0; i < this.output; i++)
            {
                output[i] += outputBiases[i];
            }

            //Activation
            switch (oType)
            {
                case ActivationType.HeavisideStep:
                    for (int i = 0; i < this.output; i++)
                        output[i] = ActivationMethods.HeavisideStep(output[i]);
                    break;
                case ActivationType.HyperbolicTangent:
                    for (int i = 0; i < this.output; i++)
                        output[i] = ActivationMethods.HyperbolicTangent(output[i]);
                    break;
                case ActivationType.LogisticSigmoid:
                    for (int i = 0; i < this.output; i++)
                        output[i] = ActivationMethods.LogisticSigmoid(output[i]);
                    break;
                case ActivationType.Softmax:
                    output = ActivationMethods.Softmax(output);
                    break;
            }
            #endregion

            return output;
        }

        /// <summary>
        /// Get weights and biases
        /// </summary>
        /// <returns>a double array of data containning weights and biases</returns>
        public double[] GetWeights()
        {
            int len = WeightsLength;
            double[] result = new double[len];

            int k = 0;

            #region input->hidden
            //Weights
            for (int i = 0; i < this.input; i++)
            {
                for (int j = 0; j < this.hidden; j++)
                {
                    result[k] = ihweights[i][j];
                    k++;
                }
            }

            //Biases
            for (int i = 0; i < this.hidden; i++)
            {
                result[k] = hiddenBiases[i];
                k++;
            }
            #endregion

            #region hidden->output
            //Weights
            for (int i = 0; i < this.hidden; i++)
            {
                for (int j = 0; j < this.output; j++)
                {
                    result[k] = howeights[i][j];
                    k++;
                }
            }

            //Biases
            for (int i = 0; i < this.output; i++)
            {
                result[k] = outputBiases[i];
                k++;
            }
            #endregion

            return result;
        }

        /// <summary>
        /// Set weights and biases
        /// </summary>
        /// <param name="data">an double array containning weights and biases NOTE: array length must equal weights length + biases length</param>
        public void SetWeights(double[] data)
        {
            int len = WeightsLength;
            if (len != data.Length)
                throw new Exception("Data Length doesn't match.");

            int k = 0;

            #region input->hidden
            //Weights
            for (int i = 0; i < this.input; i++)
            {
                for (int j = 0; j < this.hidden; j++)
                {
                    ihweights[i][j] = data[k];
                    k++;
                }
            }

            //Biases
            for (int i = 0; i < this.hidden; i++)
            {
                hiddenBiases[i] = data[k];
                k++;
            }
            #endregion

            #region hidden->output
            //Weights
            for (int i = 0; i < this.hidden; i++)
            {
                for (int j = 0; j < this.output; j++)
                {
                    howeights[i][j] = data[k];
                    k++;
                }
            }

            //Biases
            for (int i = 0; i < this.output; i++)
            {
                outputBiases[i] = data[k];
                k++;
            }
            #endregion
        }

        /// <summary>
        /// Generates Random values for weights and biases
        /// </summary>
        public void GenerateWeights()
        {
            Random rnd = new Random();
            int len = WeightsLength;
            double[] initialWeights = new double[len];
            double lo = -0.01;
            double hi = 0.01;
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (hi - lo) * rnd.NextDouble() + lo;
            SetWeights(initialWeights);
        }

        /// <summary>
        /// Checks how many test data it gets right
        /// </summary>
        /// <param name="data">list of test data</param>
        /// <returns>percentage of acuracy</returns>
        public double Accuracy(double[][] data)
        {
            if (data[0].Length != this.Input + this.Output)
                throw new Exception("Array input and outputs don't match");

            double correct = 0;
            double wrong = 0;

            for (int i = 0; i < data.Length; i++)
            {
                int k = 0;
                double[] inputs = new double[this.Input];
                double[] targets = new double[this.Output];

                for (int j = 0; j < this.Input; j++)
                    inputs[j] = data[i][k++];
                for (int j = 0; j < this.Output; j++)
                    targets[j] = data[i][k++];

                double[] outputs = ComputeResults(inputs);

                int targetVal = 0;
                int outputVal = 0;

                for (int j = 0; j < targets.Length; j++)
                    if (targets[j] > targets[targetVal])
                        targetVal = j;

                for (int j = 0; j < outputs.Length; j++)
                    if (outputs[j] > outputs[outputVal])
                        outputVal = j;


                if (targetVal == outputVal)
                    correct++;
                else
                    wrong++;
            }

            return correct / (correct + wrong);
        }

        #endregion

        #region Private/Protected Methods

        protected static int[] Shuffle(int[] sequence)
        {
            Random rnd = new Random();

            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }

            return sequence;
        }

        protected static int[] Shuffle(int[] sequence, Random rnd)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }

            return sequence;
        }

        #endregion
    }
}
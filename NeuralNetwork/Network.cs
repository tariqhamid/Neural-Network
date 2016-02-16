using System;

namespace NeuralNetwork
{
    public class Network
    {
        #region Getters & Setters

        public Neuron[][] Neurons
        {
            get
            {
                return neurons;
            }
            protected set
            {
                neurons = value;
            }
        }

        public Synapsis[][] Synapsis
        {
            get
            {
                return synapsis;
            }
            protected set
            {
                synapsis = value;
            }
        }

        #endregion

        #region Variables

        private Neuron[][] neurons;
        private Synapsis[][] synapsis;
        protected int[] layers;
        protected Random random;

        #endregion

        #region Constructors

        public Network(int[] layers, bool connectAllNodes)
        {
            if (layers.Length < 3)
                throw new Exception("There needs to be atleast 2 layers");

            this.layers = layers;

            neurons = new Neuron[layers.Length][];
            synapsis = new Synapsis[layers.Length - 1][];

            //Setup Neurons
            for (int l = 0; l < layers.Length; l++)
            {
                neurons[l] = new Neuron[layers[l]];
                for (int i = 0; i < layers[l]; i++)
                {
                    neurons[l][i] = new Neuron(ActivationType.HyperbolicTangent, Convert.ToSingle(random.NextDouble()), Convert.ToSingle(random.NextDouble()));
                }
            }

            //Setup Synapsis
            for (int i = 0; i < layers.Length - 1; i++)
            {
                AddConnection(i, i + 1, connectAllNodes);
            }
        }

        public Network(int[] layers, ActivationType[] activationMethods, bool connectAllNodes)
        {
            if (layers.Length < 3)
                throw new Exception("There needs to be atleast 2 layers");

            if (layers.Length != activationMethods.Length)
                throw new Exception("Activation Methods length doesn't equal layers length");

            this.layers = layers;

            neurons = new Neuron[layers.Length][];
            synapsis = new Synapsis[layers.Length - 1][];

            //Setup Neurons
            for (int l = 0; l < layers.Length; l++)
            {
                neurons[l] = new Neuron[layers[l]];
                for (int i = 0; i < layers[l]; i++)
                {
                    neurons[l][i] = new Neuron(activationMethods[l], Convert.ToSingle(random.NextDouble()), Convert.ToSingle(random.NextDouble()));
                }
            }

            //Setup Synapsis
            for (int i = 0; i < layers.Length - 1; i++)
            {
                AddConnection(i, i + 1, connectAllNodes);
            }
        }

        #endregion

        #region Public Methods

        public float[] Compute(float[] xValues)
        {
            if (xValues.Length != layers[0])
                throw new Exception("X Values length doesn't match amount of input nodes");

            //Setup Input Layer
            for (int i = 0; i < xValues.Length; i++)
            {
                neurons[0][i].charge = xValues[i];
            }

            //Go Through Each Layer
            for (int i = 0; i < layers.Length - 1; i++)
            {
                float[] charges = new float[layers[i + 1]];
                int synCounter = 0;
                //Compute Weights
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    for (int k = 0; k < neurons[i + 1].Length; k++)
                    {
                        charges[k] += synapsis[i][synCounter].weight * neurons[i][j].charge;
                        synCounter++;
                    }
                }
                //Add Bias, Activation Function & Apply results to neuron
                for (int j = 0; j < neurons[i + 1].Length; j++)
                {
                    charges[j] += neurons[i + 1][j].bias;
                    charges = ActivationMethods.Activation(charges, neurons[i + 1][j].activation);
                    neurons[i + 1][j].charge = charges[j];
                }
            }

            //Get Output Layer
            float[] yValues = new float[neurons[neurons.Length - 1].Length];
            for (int i = 0; i < yValues.Length; i++)
            {
                yValues[i] = neurons[neurons.Length - 1][i].charge;
            }

            return yValues;
        }

        #endregion

        #region Private Methods

        private void AddConnection(int startLayer, int endLayer, bool connectAllNodes)
        {
            synapsis[startLayer] = new Synapsis[layers[startLayer] * layers[endLayer]];
            int synCounter = 0;
            for (int i = 0; i < neurons[startLayer].Length; i++)
            {
                for (int j = 0; j < neurons[endLayer].Length; j++)
                {
                    synapsis[startLayer][synCounter] = new Synapsis(i, j, Convert.ToSingle(random.NextDouble()));
                    if (!connectAllNodes)
                    {
                        //Set weights to 0 for some synapsis
                    }
                    synCounter++;
                }
            }
        }

        #endregion
    }
}
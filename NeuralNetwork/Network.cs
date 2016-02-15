using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class Network
    {
        public Neuron[][] neurons;
        public Synapsis[][] synapsis;
        private int[] layers;

        public Network(int[] layers)
        {
            this.layers = layers;

            neurons = new Neuron[layers.Length][];
            synapsis = new Synapsis[layers.Length - 1][];

            //Setup Neurons
            for (int l = 0; l < layers.Length; l++)
            {
                neurons[l] = new Neuron[layers[l]];
                for (int i = 0; i < layers[l]; i++)
                {
                    neurons[l][i] = new Neuron();
                }
            }

            //Setup Synapsis
            for (int i = 0; i < layers.Length - 1; i++)
            {
                AddConnection(i, i + 1, true);
            }
        }

        public float[] Compute(float[] xValues)
        {
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
                        charges[k] += synapsis[i][k].weight * neurons[i][j].charge;
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

        private void AddConnection(int startLayer, int endLayer, bool connectAllNodes)
        {
            synapsis[startLayer] = new Synapsis[layers[startLayer] * layers[endLayer]];
            int synCounter = 0;
            for (int i = 0; i < neurons[startLayer].Length; i++)
            {
                for (int j = 0; j < neurons[endLayer].Length; j++)
                {
                    synapsis[startLayer][synCounter] = new Synapsis(i, j);
                    if (!connectAllNodes)
                    {
                        //Set weights to 0 for some synapsis
                    }
                    synCounter++;
                }
            }
        }
    }
}
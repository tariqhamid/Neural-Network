using System;
using System.Collections.Generic;
using System.Text;

namespace NN
{
    /// <summary>
    /// Uses Particle Swarm Optimisation To Train Neural Network
    /// Particle Swarm Optimisation is faster then genetic algorithm and slower then back propagation.
    /// Generally the best technique to train neural network.
    /// </summary>
    public class ParticleSwarmOptimisation : NeuralNetwork
    {
        #region Varibles

        /// <summary>
        /// Uses Back Propagation To Train Neural Network
        /// </summary>
        private class Particle
        {
            public double[] position; // equivalent to NN weights
            public double error; // measure of fitness
            public double[] velocity;

            public double[] bestPosition; // by this Particle
            public double bestError;

            public Particle(double[] position, double error, double[] velocity, double[] bestPosition, double bestError)
            {
                this.position = position;
                this.error = error;
                this.velocity = velocity;
                this.bestPosition = bestPosition;
                this.bestError = bestError;
            }
        }

        #endregion

        #region Constructors

        /// <summary>
        /// Initialise Neural Network and Back Propagation learning
        /// </summary>
        /// <param name="input">Length of input layer</param>
        /// <param name="hidden">Length of hidden layer</param>
        /// <param name="output">Length of output layer</param>
        /// <param name="hType">which type of activation method would be used in hidden</param>
        /// <param name="oType">which type of activation method would be used in output</param>
        public ParticleSwarmOptimisation(int input, int hidden, int output, ActivationType hType, ActivationType oType)
            : base(input, hidden, output, hType, oType)
        {

        }

        /// <summary>
        /// Initialise Neural Network and Back Propagation learning
        /// </summary>
        /// <param name="nn">Use Existing Neural Network</param>
        public ParticleSwarmOptimisation(NeuralNetwork nn)
            :base(nn.Input, nn.Hidden, nn.Output, nn.Hidden_Type, nn.Output_Type)
        {

        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Use this method to train the neural network
        /// </summary>
        /// <param name="data">A array containning the input and output for the neural network to learn</param>
        /// <param name="particles">How many particles there should be for trainning</param>
        /// <param name="exitError">If the error is as low as this value the trainning will stop</param>
        /// <param name="deathProbability">Death Probability</param>
        /// <param name="repeat">How many times would you like to repeat the trainning data</param>
        public void Train(double[][] data, int particles, double exitError, double deathProbability, int repeat)
        {
            Random rnd = new Random(16);
            int numWeights = WeightsLength;
            int r = 0;
            double minX = -10.0;
            double maxX = 10.0;
            double w = 0.729; // inertia weight
            double c1 = 1.49445; // cognitive weight
            double c2 = 1.49445; // social weight
            double r1, r2; // randomizations
            Particle[] swarm = new Particle[particles];
            double[] bestGlobalPosition = new double[numWeights];
            double bestGlobalError = double.MaxValue;

            for (int i = 0; i < swarm.Length; ++i)
            {
                double[] randomPosition = new double[numWeights];
                for (int j = 0; j < randomPosition.Length; ++j)
                {
                    randomPosition[j] = (maxX - minX) *
                      rnd.NextDouble() + minX;
                }
                double error = MeanSquaredError(data, randomPosition);

                double[] randomVelocity = new double[numWeights];
                for (int j = 0; j < randomVelocity.Length; ++j)
                {
                    double lo = 0.1 * minX;
                    double hi = 0.1 * maxX;
                    randomVelocity[j] = (hi - lo) *
                      rnd.NextDouble() + lo;
                }
                swarm[i] = new Particle(randomPosition, error, randomVelocity, randomPosition, error);
                if (swarm[i].error < bestGlobalError)
                {
                    bestGlobalError = swarm[i].error;
                    swarm[i].position.CopyTo(bestGlobalPosition, 0);
                }
            }

            int[] sequence = new int[particles];
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            while (r < repeat)
            {
                if (bestGlobalError < exitError) break;
                sequence = Shuffle(sequence, rnd); // randomize particle order
                double[] newVelocity = new double[numWeights];
                double[] newPosition = new double[numWeights];
                double newError; // step 3
                for (int pi = 0; pi < swarm.Length; ++pi)
                {
                    int i = sequence[pi];
                    Particle currP = swarm[i];
                    for (int j = 0; j < currP.velocity.Length; ++j)
                    {
                        r1 = rnd.NextDouble();
                        r2 = rnd.NextDouble();
                        newVelocity[j] = (w * currP.velocity[j]) +
                         (c1 * r1 * (currP.bestPosition[j] - currP.position[j])) +
                          (c2 * r2 * (bestGlobalPosition[j] - currP.position[j]));
                    }
                    newVelocity.CopyTo(currP.velocity, 0);

                    for (int j = 0; j < currP.position.Length; ++j)
                    {
                        newPosition[j] = currP.position[j] + newVelocity[j];
                        if (newPosition[j] < minX) // keep in range

                            newPosition[j] = minX;
                        else if (newPosition[j] > maxX)
                            newPosition[j] = maxX;
                    }
                    newPosition.CopyTo(currP.position, 0);
                    newError = MeanSquaredError(data, newPosition);
                    currP.error = newError;
                    if (newError < currP.bestError) // particle best?
                    {
                        newPosition.CopyTo(currP.bestPosition, 0);
                        currP.bestError = newError;
                    }
                    if (newError < bestGlobalError) // global best?
                    {
                        newPosition.CopyTo(bestGlobalPosition, 0);
                        bestGlobalError = newError;
                    }
                    double die = rnd.NextDouble();
                    if (die < deathProbability)
                    {
                        // new position, leave velocity, update error
                        for (int j = 0; j < currP.position.Length; ++j)
                            currP.position[j] = (maxX - minX) * rnd.NextDouble() + minX;
                        currP.error = MeanSquaredError(data, currP.position);
                        currP.position.CopyTo(currP.bestPosition, 0);
                        currP.bestError = currP.error;

                        if (currP.error < bestGlobalError) // global best?
                        {
                            bestGlobalError = currP.error;
                            currP.position.CopyTo(bestGlobalPosition, 0);
                        }
                    }
                }
                Console.Write("\rLearning: " + (r + 1) + "/" + repeat);
                r++;
            }
            Console.WriteLine();
            if (r != repeat)
                Console.WriteLine("Stopping learning because error is too low");

            this.SetWeights(bestGlobalPosition);
            double[] retResult = new double[numWeights];
            Array.Copy(bestGlobalPosition, retResult, retResult.Length);
        }

        #endregion

        #region Private Methods

        private double MeanSquaredError(double[][] data, double[] weights)
        {
            this.SetWeights(weights);

            double[] xValues = new double[this.Input]; // inputs
            double[] tValues = new double[this.Output]; // targets
            double sumSquaredError = 0.0;
            for (int i = 0; i < data.Length; ++i)
            {
                // assumes data has x-values followed by y-values
                Array.Copy(data[i], xValues, this.Input);
                Array.Copy(data[i], this.Input, tValues, 0, this.Output);
                double[] yValues = this.ComputeResults(xValues);
                for (int j = 0; j < yValues.Length; ++j)
                    sumSquaredError += ((yValues[j] - tValues[j]) * (yValues[j] - tValues[j]));
            }
            return sumSquaredError / data.Length;
        }

        #endregion
    }
}
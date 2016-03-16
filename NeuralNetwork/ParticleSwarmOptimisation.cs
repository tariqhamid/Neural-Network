using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public delegate float AccuracyCallBack(float[] data, float[] weights);

    public class ParticleSwarmOptimisation
    {
        private AccuracyCallBack accuracy;
        private float[] weights;
        public struct Particle
        {
            public float[] position;
            public float[] velocity;

            public float[] bestPosition;
            public float error;
            public float bestError;

            public Particle(float[] position, float[] velocity, float[] bestPosition, float error, float bestError)
            {
                this.position = position;
                this.velocity = velocity;
                this.bestPosition = bestPosition;
                this.error = error;
                this.bestError = bestError;
            }
        }

        public ParticleSwarmOptimisation(AccuracyCallBack accuracy, float[] weights)
        {
            this.accuracy = accuracy;
            this.weights = weights;
        }

        public float[] Train(float[] data, int particles, float exitError, float deathProbability, int repeat)
        {
            if (accuracy == null)
                throw new Exception("Cannot continue without a way to calculate how bit the error is");
            if (weights.Length < 1)
                throw new Exception("Weight length is less then 1");

            Random rnd = new Random(16);
            int   r    =  0;
            float minX = -10.0f;
            float maxX =  10.0f;
            float w    =  0.729f;   //Inertia weight
            float c1   =  1.49445f; //Cognitive weight
            float c2   =  1.49445f; //Social weight
            float r1, r2;           //Randomizations
            Particle[] swarm = new Particle[particles];
            float[] bestGlobalPosition = new float[weights.Length];
            float bestGlobalError = float.MaxValue;

            //Create a bunch of particles with random position & velocity
            for (int i = 0; i < swarm.Length; ++i)
            {
                float[] randomPosition = new float[weights.Length];
                for (int j = 0; j < randomPosition.Length; j++)
                {
                    randomPosition[j] = (maxX - minX) * Convert.ToSingle(rnd.NextDouble()) + minX;
                }
                //Divide accuracy by 1 to get error
                float error = (1 / accuracy(data, weights));

                float[] randomVelocity = new float[weights.Length];
                for (int j = 0; j < randomVelocity.Length; j++)
                {
                    float low  = 0.1f * minX;
                    float high = 0.1f * maxX;
                    randomVelocity[j] = (high - low) * Convert.ToSingle(rnd.NextDouble()) + low;
                }
                swarm[i] = new Particle(randomPosition, randomVelocity, randomPosition, error, error);
                if (swarm[i].error < bestGlobalError)
                {
                    bestGlobalError = swarm[i].error;
                    swarm[i].position.CopyTo(bestGlobalPosition, 0);
                }
            }

            //Create a sequence for particles so we can shufle them later on
            int[] sequence = new int[particles];
            for (int i = 0; i < sequence.Length; i++)
                sequence[i] = i;

            while (r < repeat)
            {
                if (bestGlobalError < exitError)
                    break;

                //Randomize Particle Order
                sequence = Shuffle(sequence, rnd);
                float[] newVelocity = new float[weights.Length];
                float[] newPosition = new float[weights.Length];
                float newError;

                for (int p = 0; p < swarm.Length; p++)
                {
                    int i = sequence[p];

                    //Calculate velocity
                    for (int j = 0; j < swarm[i].velocity.Length; j++)
                    {
                        r1 = Convert.ToSingle(rnd.NextDouble());
                        r2 = Convert.ToSingle(rnd.NextDouble());
                        newVelocity[j] = (w * swarm[i].velocity[j]) +
                            (c1 * r1 * (swarm[i].bestPosition[j] - swarm[i].position[j])) +
                            (c2 * r2 * (bestGlobalPosition[j] - swarm[i].position[j]));
                    }
                    swarm[i].velocity = newVelocity;

                    //Calculate Position
                    for (int j = 0; j < swarm[i].position.Length; j++)
                    {
                        newPosition[j] = swarm[i].position[j] + newVelocity[j];
                        //Keep in Range
                        if (newPosition[j] < minX)
                            newPosition[j] = minX;
                        else if (newPosition[j] > maxX)
                            newPosition[j] = maxX;
                    }
                    swarm[i].position = newPosition;

                    //Get Best Error/Position
                    newError = (1 / accuracy(data, swarm[i].position));
                    swarm[i].error = newError;
                    //Particle Best
                    if (newError < swarm[i].bestError)
                    {
                        swarm[i].bestPosition = newPosition;
                    }
                    //Global Best
                    if (newError < bestGlobalError)
                    {
                        bestGlobalPosition = newPosition;
                        bestGlobalError = newError;
                    }

                    //Use Particle Death Probability And Randomly Kill It
                    float die = Convert.ToSingle(rnd.NextDouble());
                    if (die < deathProbability)
                    {
                        //New Position, Leave Velocity, Update Error
                        for (int j = 0; j < swarm[i].position.Length; j++)
                        {
                            swarm[i].position[j] = (maxX - minX) * Convert.ToSingle(rnd.NextDouble()) + minX;
                        }
                        swarm[i].error = (1 / accuracy(data, swarm[i].position));
                        swarm[i].bestPosition = swarm[i].position;

                        //Global Best
                        if (swarm[i].error < bestGlobalError)
                        {
                            bestGlobalError = swarm[i].error;
                            bestGlobalPosition = swarm[i].position;
                        }
                    }
                }


                r++;
            }

            return bestGlobalPosition;
        }

        private static int[] Shuffle(int[] sequence, Random rnd)
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
    }
}
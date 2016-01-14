using System;
using System.Collections.Generic;
using System.Text;

namespace NN
{
    /// <summary>
    /// Different types of activation method that can be used.
    /// </summary>
    public enum ActivationType
    {
        /// <summary>
        /// Do not recomend this. It skips the activation step
        /// </summary>
        None,
        /// <summary>
        /// Uses logistic sigmoid gives values between 0  and 1
        /// </summary>
        LogisticSigmoid,
        /// <summary>
        /// Hyperbolic tangent gives values between -1 and 1
        /// </summary>
        HyperbolicTangent,
        /// <summary>
        /// Heaviside step gives either a vlue of 0 or 1
        /// </summary>
        HeavisideStep,
        /// <summary>
        /// Softmax's gives values between 0 and 1. These values also sum up to 1
        /// </summary>
        Softmax
    }

    /// <summary>
    /// Contains activation methods for neural networks
    /// </summary>
    public class ActivationMethods
    {
        #region Activation Methods

        /// <summary>
        /// Apply's softmax activation to a list of values
        /// </summary>
        /// <param name="values">list of values needed to be activated</param>
        /// <returns>returns the updated values</returns>
        public static double[] Softmax(double[] values)
        {
            // determine max output sum
            // does all output nodes at once so scale doesn't have to be re-computed each time
            double max = values[0];
            for (int i = 0; i < values.Length; ++i)
                if (values[i] > max) max = values[i];

            // determine scaling factor -- sum of exp(each val - max)
            double scale = 0.0;
            for (int i = 0; i < values.Length; ++i)
                scale += Math.Exp(values[i] - max);

            double[] result = new double[values.Length];
            for (int i = 0; i < values.Length; ++i)
                result[i] = Math.Exp(values[i] - max) / scale;

            return result; // now scaled so that xi sum to 1.0

        }

        /// <summary>
        /// Apply's hyperbolic tangent activation to a value
        /// </summary>
        /// <param name="value">The value that needs to get activation method applied to it</param>
        /// <returns>Converted value after appling the activation method</returns>
        public static double HyperbolicTangent(double value)
        {
            if (value < -20.0) return -1.0; // approximation is correct to 30 decimals
            else if (value > 20.0) return 1.0;
            else return Math.Tanh(value);
        }

        /// <summary>
        /// Apply's logistic sigmoid activation to a value
        /// </summary>
        /// <param name="value">The value that needs to get activation method applied to it</param>
        /// <returns>Converted value after applying the activation method</returns>
        public static double LogisticSigmoid(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        /// <summary>
        /// Apply's heavisidestep activation to a value. This is not supported by back propagation.
        /// </summary>
        /// <param name="values">The value that needs to get activation method applied to it</param>
        /// <returns>Converted value after applying the activation method</returns>
        public static double HeavisideStep(double values)
        {
            if (values < 0)
                return 0;
            else
                return 1;
        }

        #endregion

        #region Inverse Activation Methods

        /// <summary>
        /// Removes logistic sigmoid activation to a value
        /// </summary>
        /// <param name="value">The value that needs to get activation method removed to it</param>
        /// <returns>Converted value after removing the activation method</returns>
        public static double LogisticSigmoidDerivative(double value)
        {
            return LogisticSigmoid(value) * (1 - LogisticSigmoid(value));
        }

        /// <summary>
        /// Removes Hyperbolic tangent activation to a value
        /// </summary>
        /// <param name="value">The value that needs to get activation method removed to it</param>
        /// <returns>Conerted value after removing the activation method</returns>
        public static double HyperbolicTangentDerivative(double value)
        {
            return 1 - Math.Pow(HyperbolicTangent(value), 2);
        }

        #endregion
    }
}

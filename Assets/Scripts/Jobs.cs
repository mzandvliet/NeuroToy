using Unity.Jobs;
using Unity.Collections;
using static Unity.Mathematics.math;
using UnityEngine;

/* Todo:
- Reference types not allowed in jobs, so we can't pass along a System.Random instance.
So immediately we get to the interesting challenge of generating random numbers for Burst.

- After checking out singlethreaded performance, try writing ParallalFor variants of
many of these functions, since most calculations are independent.

- Maybe we can use NativeSlice concept to flatten structures into 1d arrays
 */

namespace NeuralJobs {
    public static class JobMath {
        // Quick and dirty, probably not great
        public static float Gaussian(MTRandom random, float mean, float stddev) {
            float x1 = 1f - random.NextFloat();
            float x2 = 1f - random.NextFloat();

            float y1 = sqrt(-2.0f * log(x1)) * cos(2.0f * Mathf.PI * x2);
            return y1 * stddev + mean;
        }

        public static float Sigmoid(float x) {
            return 1f / (1f + exp(-x));
        }
    }

    public struct CopyToJob : IJob {
        public NativeArray<float> A;
        public NativeArray<float> T;

        public void Execute() {
            if (A.Length != T.Length) {
                Debug.LogError("Arrays need to be of same length.");
                return;
            }

            for (int i = 0; i < A.Length; i++) {
                T[i] = A[i];
            }
        }
    }

    // public struct GaussianJob : IJob {
    //     [ReadOnly] public MTRandom Random;
    //     [ReadOnly] public float Mean;
    //     [ReadOnly] public float Std;
    //     [WriteOnly] public NativeArray<float> T;

    //     public void Execute() {
    //         for (int i = 0; i < T.Length; i++) {
    //             T[i] = JobMath.Gaussian(Random, Mean, Std);
    //         }
    //     }
    // }

    public struct SigmoidJob : IJob {
        public NativeArray<float> A;

        public void Execute() {
            for (int i = 0; i < A.Length; i++) {
                A[i] = JobMath.Sigmoid(A[i]);
            }
        }
    }

    // Calculates transpose(weights) * inputs
    public struct DotJob : IJob {
        [ReadOnly] public NativeArray<float> Input;
        [ReadOnly] public NativeArray<float> Weights;
        public NativeArray<float> Output;

        public void Execute() {
            // Outer loop is parallelizable
            for (int n = 0; n < Output.Length; n++) {
                // Inner loop is best kep synchronous
                for (int i = 0; i < Input.Length; i++) {
                    Output[n] += Input[i] * Weights[Input.Length * n + i];
                }
                Output[n] /= (float)Input.Length;
            }
        }
    }
}

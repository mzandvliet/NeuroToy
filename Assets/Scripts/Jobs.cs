using Unity.Jobs;
using Unity.Collections;
using static Unity.Mathematics.math;
using UnityEngine;
using Unity.Burst;

/* Todo:
- Reference types not allowed in jobs, so we can't pass along a System.Random instance.
So immediately we get to the interesting challenge of generating random numbers for Burst.

- After checking out singlethreaded performance, try writing ParallalFor variants of
many of these functions, since most calculations are independent.

- Maybe we can use NativeSlice concept to flatten structures into 1d arrays

- [ComputeJobOptimization]
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

        public static float SigmoidPrime(float sigmoidX) {
            return sigmoidX * (1f - sigmoidX);
        }
    }

    [BurstCompile]
    public struct CopySubsetJob : IJob {
        [ReadOnly] public NativeArray<float> A;
        [WriteOnly] public NativeArray<float> B;
        [ReadOnly] public int AStart;
        [ReadOnly] public int ALength;
        [ReadOnly] public int BStart;

        public void Execute() {
            for (int i = 0; i < ALength; i++) {
                B[BStart + i] = A[AStart + i];
            }
        }
    }

    [BurstCompile]
    public struct CopySubsetParallelJob : IJobParallelFor {
        [ReadOnly] public NativeArray<float> A;
        [WriteOnly] public NativeArray<float> B;
        [ReadOnly] public int AStart;
        [ReadOnly] public int ALength;
        [ReadOnly] public int BStart;

        public void Execute(int i) {
            B[BStart + i] = A[AStart + i];
        }
    }

    [BurstCompile]
    public struct CopyJob : IJob {
        [ReadOnly] public NativeArray<float> From;
        [WriteOnly] public NativeArray<float> To;

        public void Execute() {
            // if (A.Length != T.Length) {
            //     Debug.LogError("Arrays need to be of same length.");
            //     return;
            // }

            for (int i = 0; i < From.Length; i++) {
                To[i] = From[i];
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

    [BurstCompile]
    public struct SigmoidEqualsJob : IJob {
        public NativeArray<float> A;

        public void Execute() {
            for (int i = 0; i < A.Length; i++) {
                A[i] = JobMath.Sigmoid(A[i]);
            }
        }
    }

    [BurstCompile]
    public struct AddEqualsJob : IJob {
        [ReadOnly] public NativeArray<float> A;
        public NativeArray<float> B;

        public void Execute() {
            for (int i = 0; i < A.Length; i++) {
                B[i] += A[i];
            }
        }
    }

    [BurstCompile]
    public struct SubtractEqualsJob : IJob {
        [ReadOnly] public NativeArray<float> A;
        public NativeArray<float> B;

        public void Execute() {
            // if (A.Length != B.Length || A.Length != T.Length) {
            //     Debug.LogError("Arrays need to be of same length.");
            //     return;
            // }

            for (int i = 0; i < A.Length; i++) {
                B[i] -= A[i];
            }
        }
    }

    [BurstCompile]
    public struct MultiplyEqualsJob : IJob {
        public NativeArray<float> A;
        [ReadOnly] public float Scalar;

        public void Execute() {
            // if (A.Length != B.Length || A.Length != T.Length) {
            //     Debug.LogError("Arrays need to be of same length.");
            //     return;
            // }

            for (int i = 0; i < A.Length; i++) {
                A[i] *= Scalar;
            }
        }
    }

    [BurstCompile]
    public struct SubtractJob : IJob {
        [ReadOnly] public NativeArray<float> A;
        [ReadOnly] public NativeArray<float> B;
        public NativeArray<float> T;

        public void Execute() {
            // if (A.Length != B.Length || A.Length != T.Length) {
            //     Debug.LogError("Arrays need to be of same length.");
            //     return;
            // }

            for (int i = 0; i < A.Length; i++) {
                T[i] = A[i] - B[i];
            }
        }
    }

    // Calculates transpose(weights) * inputs
    [BurstCompile]
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

    // Todo: The way backwards passes are written needs lots of restructuring
    [BurstCompile]
    public struct BackwardsFinalJob : IJob {
        [ReadOnly] public NativeArray<float> DCDO;
        [ReadOnly] public NativeArray<float> OutputsPrev;
        [ReadOnly] public NativeArray<float> Output;
        public NativeArray<float> DCDZ;
        public NativeArray<float> DCDW;

        public void Execute() {
            for (int n = 0; n < Output.Length; n++) {
                float dOdZ = JobMath.SigmoidPrime(Output[n]); // Reuses forward pass evaluation of act(z)
                DCDZ[n] = DCDO[n] * dOdZ;

                for (int w = 0; w < OutputsPrev.Length; w++) {
                    DCDW[n * Output.Length + w] = DCDZ[n] * OutputsPrev[w];
                }
            }
        }
    }

    [BurstCompile]
    public struct BackwardsJob : IJob {
        [ReadOnly] public NativeArray<float> DCDZNext;
        [ReadOnly] public NativeArray<float> WeightsNext;
        [ReadOnly] public NativeArray<float> OutputsPrev;
        [ReadOnly] public NativeArray<float> Output;
        public NativeArray<float> DCDZ;
        public NativeArray<float> DCDW;

        public void Execute() {
            for (int n = 0; n < Output.Length; n++) {
                float dOdZ = JobMath.SigmoidPrime(Output[n]);

                DCDZ[n] = 0f;
                for (int nNext = 0; nNext < DCDZNext.Length; nNext++) {
                    DCDZ[n] += DCDZNext[nNext] * WeightsNext[nNext * DCDZNext.Length + n];
                }
                DCDZ[n] *= dOdZ;

                for (int w = 0; w < OutputsPrev.Length; w++) {
                    DCDW[n * Output.Length + w] = DCDZ[n] * OutputsPrev[w];
                }
            }
        }
    }
}

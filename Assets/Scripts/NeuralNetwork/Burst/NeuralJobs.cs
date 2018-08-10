using Unity.Jobs;
using Unity.Collections;
using static Unity.Mathematics.math;
using UnityEngine;
using Unity.Burst;

/* Todo:
Always flush schedule batches
When you want your jobs to start, you need to flush the scheduled batch with JobHandle.ScheduleBatchedJobs. Not doing this delays the scheduling until another job waits for the result.

- Reference types not allowed in jobs, so we can't pass along a System.Random instance.
So immediately we get to the interesting challenge of generating random numbers for Burst.

- Basic error handling, such as when input array lengths mismatch (can be done outside of job system)
    - Check that all job inputs are set to valid state

- Try writing ParallalFor variants of many of these functions, since most calculations are independent.

- Maybe we can use NativeSlice concept to flatten structures into 1d arrays

- Parameterize activation functions somehow, without making jobs slow through indirection

 */

namespace NNBurst {
    public static class NeuralMath {
        // Quick and dirty, probably not great
        public static float Gaussian(MTRandom random, float mean, float stddev) {
            float x1 = 1f - random.NextFloat();
            float x2 = 1f - random.NextFloat();

            float y1 = sqrt(-2.0f * log(x1)) * cos(2.0f * Mathf.PI * x2);
            return y1 * stddev + mean;
        }

        #region Activation Functions

        public static float Sigmoid(float x) {
            return 1f / (1f + exp(-x));
        }

        public static float SigmoidPrime(float sigmoidX) {
            return sigmoidX * (1f - sigmoidX);
        }

        public static float ReLU(float x) {
            return max(x, 0f);
        }

        public static float ReLUPrime(float reluX) {
            return reluX > 0f ? 1f : 0f;
        }

        #endregion

        #region NativeArray Math

        public static void RandomGaussian(System.Random random, NativeArray<float> values, float mean, float std) {
            for (int i = 0; i < values.Length; i++) {
                values[i] = NNClassic.Utils.Gaussian(random, mean, std);
            }
        }

        public static void Subtract(NativeArray<float> a, NativeArray<float> b, NativeArray<float> result) {
            if (a.Length != b.Length) {
                throw new System.ArgumentException("Lengths of arrays have to match");
            }

            for (int i = 0; i < a.Length; i++) {
                result[i] = a[i] - b[i];
            }
        }

        public static void ClassToOneHot(int c, NativeArray<float> vector) {
            for (int i = 0; i < vector.Length; i++) {
                vector[i] = i == c ? 1f : 0f;
            }
        }

        public static int ArgMax(NativeArray<float> data) {
            float largestActivation = float.MinValue;
            int idx = 0;
            for (int i = 0; i < data.Length; i++) {
                if (data[i] > largestActivation) {
                    largestActivation = data[i];
                    idx = i;
                }
            }
            return idx;
        }

        public static float Cost(NativeArray<float> vector) {
            float sum = 0f;
            for (int i = 0; i < vector.Length; i++) {
                sum += vector[i] * vector[i];
            }
            return Unity.Mathematics.math.sqrt(sum);
        }

        public static bool IsEven(float x) {
            return frac(x / 2f) < 0.01f;
        }

        #endregion
    }

    public static class NeuralJobs {
        public static JobHandle CopyInput(NativeArray<float> inputs, NNBurst.Mnist.Dataset set, int imgIdx, JobHandle handle = new JobHandle()) {
            var copyInputJob = new CopySubsetJob();
            copyInputJob.From = set.Images;
            copyInputJob.To = inputs;
            copyInputJob.Length = set.ImgDims;
            copyInputJob.FromStart = imgIdx * set.ImgDims;
            copyInputJob.ToStart = 0;
            return copyInputJob.Schedule(handle);
        }

        public static JobHandle CopyInput(NativeArray<float> inputs, NNBurst.Cifar.Dataset set, int imgIdx, JobHandle handle = new JobHandle()) {
            var copyInputJob = new CopySubsetJob();
            copyInputJob.From = set.Images;
            copyInputJob.To = inputs;
            copyInputJob.Length = set.ImgDims * 3;
            copyInputJob.FromStart = imgIdx * set.ImgDims * 3;
            copyInputJob.ToStart = 0;
            return copyInputJob.Schedule(handle);
        }

        public static JobHandle ForwardPass(NativeNetwork net, NativeArray<float> input, JobHandle handle = new JobHandle()) {
            NativeArray<float> last = input;

            for (int l = 0; l < net.Layers.Length; l++) {
                var layer = net.Layers[l];

                const int numThreads = 8;

                var b = new CopyParallelJob();
                b.From = layer.Biases;
                b.To = layer.Outputs;
                handle = b.Schedule(layer.Outputs.Length, layer.Outputs.Length / numThreads, handle);

                var d = new DotParallelJob();
                d.Input = last;
                d.Weights = layer.Weights;
                d.Output = layer.Outputs;
                handle = d.Schedule(layer.Outputs.Length, layer.Outputs.Length / numThreads, handle);

                var s = new SigmoidAssignParallelJob();
                s.Data = layer.Outputs;
                handle = s.Schedule(layer.Outputs.Length, layer.Outputs.Length / numThreads, handle);

                last = layer.Outputs;
            }

            return handle;
        }

        public static JobHandle BackwardsPass(NativeNetwork net, NativeGradients gradients, NativeArray<float> input, NativeArray<float> target, JobHandle handle = new JobHandle()) {
            JobHandle h = handle;

            // Todo: make separate jobs for updating DCDZ and DCDW
            // DCDW can make good use of ParallelFor, but depends on DCDZ
            // Oh, but then, while DCDW for layer L is being calculated, DCDZ for L-1 can calculate while DCDW of L is calculating

            var subtractJob = new SubtractJob();
            subtractJob.A = net.Last.Outputs;
            subtractJob.B = target;
            subtractJob.Output = gradients.DCDO;
            h = subtractJob.Schedule(h);

            var backwardsFinalJob = new BackPropFinalJob();
            backwardsFinalJob.DCDO = gradients.DCDO;
            backwardsFinalJob.DCDZ = gradients.Last.DCDZ;
            backwardsFinalJob.DCDW = gradients.Last.DCDW;
            backwardsFinalJob.Outputs = net.Last.Outputs;
            backwardsFinalJob.OutputsPrev = net.Layers[net.Layers.Length - 2].Outputs;
            h = backwardsFinalJob.Schedule(h);

            // Note, indexing using net.layers.length here is misleading, since that count is one less than if you include input layer
            for (int l = net.Layers.Length - 2; l >= 0; l--) {
                var backwardsJob = new BackPropJob();
                backwardsJob.DCDZNext = gradients.Layers[l + 1].DCDZ;
                backwardsJob.WeightsNext = net.Layers[l + 1].Weights;
                backwardsJob.DCDZ = gradients.Layers[l].DCDZ;
                backwardsJob.DCDW = gradients.Layers[l].DCDW;
                backwardsJob.LOutputs = net.Layers[l].Outputs;
                backwardsJob.OutputsPrev = l == 0 ? input : net.Layers[l - 1].Outputs;
                h = backwardsJob.Schedule(h);
                // h = backwardsJob.Schedule(gradients.Layers[l].NumNeurons, gradients.Layers[l].NumNeurons/8, h);
            }

            return h;
        }

        public static JobHandle ZeroGradients(NativeGradients gradients, JobHandle handle = new JobHandle()) {
            // Todo: parallelize over all these independent calculations
            for (int l = 0; l < gradients.Layers.Length; l++) {
                var setBiasJob = new SetValueJob();
                setBiasJob.Data = gradients.Layers[l].DCDZ;
                setBiasJob.Value = 0f;
                var j0 = setBiasJob.Schedule(handle);

                var setWeightsJob = new SetValueJob();
                setWeightsJob.Data = gradients.Layers[l].DCDW;
                setWeightsJob.Value = 0f;
                var j1 = setWeightsJob.Schedule(handle);

                handle = JobHandle.CombineDependencies(j0, j1);
            }

            return handle;
        }

        public static JobHandle AddGradients(NativeGradients from, NativeGradients to, JobHandle handle = new JobHandle()) {
            // Todo: parallelize over layers and/or biases/weights
            for (int l = 0; l < from.Layers.Length; l++) {
                var addBiasJob = new AddAssignJob();
                addBiasJob.Data = from.Layers[l].DCDZ;
                addBiasJob.To = to.Layers[l].DCDZ;
                handle = addBiasJob.Schedule(handle);

                var addWeightsJob = new AddAssignJob();
                addWeightsJob.Data = from.Layers[l].DCDW;
                addWeightsJob.To = to.Layers[l].DCDW;
                handle = addWeightsJob.Schedule(handle);
            }

            return handle;
        }

        public static JobHandle UpdateParameters(NativeNetwork net, NativeGradients gradients, float rate, JobHandle handle = new JobHandle()) {
            // Todo: Find a nice way to fold the multiply by learning rate and addition together in one pass over the data
            // Also, parallelize over all the arrays

            for (int l = 0; l < net.Layers.Length; l++) {
                var m = new MultiplyAssignJob();
                m.Data = gradients.Layers[l].DCDZ;
                m.Value = rate;
                handle = m.Schedule(handle);

                var s = new SubtractAssignJob();
                s.Data = gradients.Layers[l].DCDZ;
                s.From = net.Layers[l].Biases;
                handle = s.Schedule(handle);

                m = new MultiplyAssignJob();
                m.Data = gradients.Layers[l].DCDW;
                m.Value = rate;
                handle = m.Schedule(handle);

                s = new SubtractAssignJob();
                s.Data = gradients.Layers[l].DCDW;
                s.From = net.Layers[l].Weights;
                handle = s.Schedule(handle);
            }

            return handle;
        }
    }

    [BurstCompile]
    public struct CopySubsetJob : IJob {
        [ReadOnly] public NativeArray<float> From;
        [WriteOnly] public NativeArray<float> To;
        [ReadOnly] public int FromStart;
        [ReadOnly] public int Length;
        [ReadOnly] public int ToStart;

        public void Execute() {
            for (int i = 0; i < Length; i++) {
                To[ToStart + i] = From[FromStart + i];
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
            for (int i = 0; i < From.Length; i++) {
                To[i] = From[i];
            }
        }
    }

    [BurstCompile]
    public struct CopyParallelJob : IJobParallelFor {
        [ReadOnly] public NativeArray<float> From;
        [WriteOnly] public NativeArray<float> To;

        public void Execute(int i) {
            To[i] = From[i];
        }
    }

    [BurstCompile]
    public struct SetValueJob : IJob {
        public NativeArray<float> Data;
        [ReadOnly] public float Value;

        public void Execute() {
            for (int i = 0; i < Data.Length; i++) {
                Data[i] = Value;
            }
        }
    }

    [BurstCompile]
    public struct SigmoidAssignJob : IJob {
        public NativeArray<float> Data;

        public void Execute() {
            for (int i = 0; i < Data.Length; i++) {
                Data[i] = NeuralMath.Sigmoid(Data[i]);
            }
        }
    }

    [BurstCompile]
    public struct SigmoidAssignParallelJob : IJobParallelFor {
        public NativeArray<float> Data;

        public void Execute(int i) {
            Data[i] = NeuralMath.Sigmoid(Data[i]);
        }
    }

    [BurstCompile]
    public struct AddAssignJob : IJob {
        [ReadOnly] public NativeArray<float> Data;
        public NativeArray<float> To;

        public void Execute() {
            for (int i = 0; i < Data.Length; i++) {
                To[i] += Data[i];
            }
        }
    }

    [BurstCompile]
    public struct SubtractAssignJob : IJob {
        [ReadOnly] public NativeArray<float> Data;
        public NativeArray<float> From;

        public void Execute() {
            for (int i = 0; i < Data.Length; i++) {
                From[i] -= Data[i];
            }
        }
    }

    [BurstCompile]
    public struct MultiplyAssignJob : IJob {
        public NativeArray<float> Data;
        [ReadOnly] public float Value;

        public void Execute() {
            for (int i = 0; i < Data.Length; i++) {
                Data[i] *= Value;
            }
        }
    }

    [BurstCompile]
    public struct SubtractJob : IJob {
        [ReadOnly] public NativeArray<float> A;
        [ReadOnly] public NativeArray<float> B;
        public NativeArray<float> Output;

        public void Execute() {
            for (int i = 0; i < A.Length; i++) {
                Output[i] = A[i] - B[i];
            }
        }
    }

    [BurstCompile]
    public struct SubtractParallelJob : IJobParallelFor {
        [ReadOnly] public NativeArray<float> A;
        [ReadOnly] public NativeArray<float> B;
        public NativeArray<float> Output;

        public void Execute(int i) {
            Output[i] = A[i] - B[i];
        }
    }

    // Calculates transpose(weights) * inputs
    [BurstCompile]
    public struct DotJob : IJob {
        [ReadOnly] public NativeArray<float> Input;
        [ReadOnly] public NativeArray<float> Weights;
        [WriteOnly] public NativeArray<float> Output;

        public void Execute() {
            // Outer loop is parallelizable
            for (int n = 0; n < Output.Length; n++) {
                float a = 0f;
                for (int i = 0; i < Input.Length; i++) {
                    a += Input[i] * Weights[Input.Length * n + i];
                }
                Output[n] = a;
            }
        }
    }

    // Calculates transpose(weights) * inputs
    [BurstCompile]
    public struct DotParallelJob : IJobParallelFor {
        [ReadOnly] public NativeArray<float> Input;
        [ReadOnly] public NativeArray<float> Weights;
        [WriteOnly] public NativeArray<float> Output;

        public void Execute(int n) {
            float a = 0f;
            for (int i = 0; i < Input.Length; i++) {
                a += Input[i] * Weights[Input.Length * n + i];
            }
            Output[n] = a;
        }
    }

    // Todo: The way backwards passes are written needs lots of restructuring
    // First, the index juggling needs to be made more readable. Like indexing
    // the higher layer's weight matrix is a mess right now.
    [BurstCompile]
    public struct BackPropFinalJob : IJob {
        [ReadOnly] public NativeArray<float> DCDO;
        [ReadOnly] public NativeArray<float> Outputs;
        [ReadOnly] public NativeArray<float> OutputsPrev;
        [WriteOnly] public NativeArray<float> DCDZ;
        [WriteOnly] public NativeArray<float> DCDW;

        public void Execute() {
            for (int n = 0; n < Outputs.Length; n++) {
                float dOdZ = NeuralMath.SigmoidPrime(Outputs[n]); // Reuses forward pass evaluation of act(z)
                float dcdzn = DCDO[n] * dOdZ;
                DCDZ[n] = dcdzn;

                for (int w = 0; w < OutputsPrev.Length; w++) {
                    DCDW[n * OutputsPrev.Length + w] = dcdzn * OutputsPrev[w];
                }
            }
        }
    }

    [BurstCompile]
    public struct BackPropJob : IJob {
        [ReadOnly] public NativeArray<float> DCDZNext;
        [ReadOnly] public NativeArray<float> WeightsNext;
        [ReadOnly] public NativeArray<float> OutputsPrev;
        [ReadOnly] public NativeArray<float> LOutputs;
        [WriteOnly] public NativeArray<float> DCDZ;
        [WriteOnly] public NativeArray<float> DCDW;

        public void Execute() {
            for (int n = 0; n < LOutputs.Length; n++) {
                float dOdZ = NeuralMath.SigmoidPrime(LOutputs[n]);

                float dcdzn = 0f;
                for (int nNext = 0; nNext < DCDZNext.Length; nNext++) {
                    dcdzn += DCDZNext[nNext] * WeightsNext[nNext * DCDZ.Length + n];
                }
                dcdzn *= dOdZ;
                DCDZ[n] = dcdzn;

                // Todo: how do we parallelize over the weights?
                // If we try to ParallelFor, compiler complains that we're writing out of bounds
                for (int w = 0; w < OutputsPrev.Length; w++) {
                    DCDW[n * OutputsPrev.Length + w] = dcdzn * OutputsPrev[w];
                }
            }
        }
    }
}

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
        public static JobHandle ForwardPass(FCNetwork net, NativeArray<float> input, JobHandle handle = new JobHandle()) {
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

        public static JobHandle BackwardsPass(FCNetwork net, FCGradients gradients, NativeArray<float> input, NativeArray<float> target, JobHandle handle = new JobHandle()) {
            JobHandle h = handle;

            // Todo: 
            // DCDW can make good use of ParallelFor, depends on DCDZ
            // Oh, but then, while DCDW for layer L is being calculated, DCDZ for L-1 can calculate while DCDW of L is calculating
            // DCDW computation is an orphan. Other computation doesn't have to wait for it.

            var sj = new SubtractJob();
            sj.A = net.Last.Outputs;
            sj.B = target;
            sj.Output = gradients.DCDO;
            h = sj.Schedule(h);

            var bfj = new BackPropSingleOutputJob();
            bfj.DCDZNext = gradients.DCDO;
            bfj.DCDZ = gradients.Last.DCDZ;
            bfj.Outputs = net.Last.Outputs;
            h = bfj.Schedule(h);

            var bwj = new BackPropWeightsJob();
            bwj.DCDZ = gradients.Last.DCDZ;
            bwj.DCDW = gradients.Last.DCDW;
            bwj.Outputs = net.Last.Outputs;
            bwj.OutputsPrev = net.Layers[net.Layers.Length - 2].Outputs;
            h = bwj.Schedule(h);

            // Note, indexing using net.layers.length here is misleading, since that count is one less than if you include input layer
            for (int l = net.Layers.Length - 2; l >= 0; l--) {
                var bej = new BackPropMultiOutputJob();
                bej.DCDZNext = gradients.Layers[l + 1].DCDZ;
                bej.WeightsNext = net.Layers[l + 1].Weights;
                bej.DCDZ = gradients.Layers[l].DCDZ;
                bej.Outputs = net.Layers[l].Outputs;
                h = bej.Schedule(h);

                bwj = new BackPropWeightsJob();
                bwj.DCDZ = gradients.Layers[l].DCDZ;
                bwj.DCDW = gradients.Layers[l].DCDW;
                bwj.Outputs = net.Layers[l].Outputs;
                bwj.OutputsPrev = l == 0 ? input : net.Layers[l - 1].Outputs;
                h = bwj.Schedule(h);
            }

            return h;
        }

        public static JobHandle ZeroGradients(FCGradients gradients, JobHandle handle = new JobHandle()) {
            // gradients = 0;

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

        public static JobHandle AddGradients(FCGradients from, FCGradients to, JobHandle handle = new JobHandle()) {
            // to.Bias += from.Bias
            // to.Weights += from.Weights

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

        public static JobHandle UpdateParameters(FCNetwork net, FCGradients gradients, float rate, JobHandle handle = new JobHandle()) {
            // Biases -= DCDZ * rate
            // Weights -= DCDW * rate

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
        [WriteOnly] public NativeArray<float> Output;

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
        [WriteOnly] public NativeArray<float> Output;

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
                for (int w = 0; w < Input.Length; w++) {
                    a += Input[w] * Weights[Input.Length * n + w];
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
            for (int w = 0; w < Input.Length; w++) {
                a += Input[w] * Weights[Input.Length * n + w];
            }
            Output[n] = a;
        }
    }

    [BurstCompile]
    public struct BackPropSingleOutputJob : IJob {
        [ReadOnly] public NativeArray<float> DCDZNext;
        [ReadOnly] public NativeArray<float> Outputs;
        [WriteOnly] public NativeArray<float> DCDZ;

        public void Execute() {
            for (int n = 0; n < Outputs.Length; n++) {
                float dOdZ = NeuralMath.SigmoidPrime(Outputs[n]);

                DCDZ[n] = DCDZNext[n] * dOdZ;
            }
        }
    }

    [BurstCompile]
    public struct BackPropMultiOutputJob : IJob {
        [ReadOnly] public NativeArray<float> DCDZNext;
        [ReadOnly] public NativeArray<float> WeightsNext;
        [ReadOnly] public NativeArray<float> Outputs;
        [WriteOnly] public NativeArray<float> DCDZ;

        public void Execute() {
            for (int n = 0; n < Outputs.Length; n++) {
                float dOdZ = NeuralMath.SigmoidPrime(Outputs[n]);

                float dcdzn = 0f;
                for (int nNext = 0; nNext < DCDZNext.Length; nNext++) {
                    dcdzn += DCDZNext[nNext] * WeightsNext[nNext * DCDZ.Length + n];
                }

                DCDZ[n] = dcdzn * dOdZ;
            }
        }
    }

    // Note: should run after calculation of DCDZ
    [BurstCompile]
    public struct BackPropWeightsJob : IJob {
        [ReadOnly] public NativeArray<float> OutputsPrev;
        [ReadOnly] public NativeArray<float> Outputs;
        [ReadOnly] public NativeArray<float> DCDZ;
        [WriteOnly] public NativeArray<float> DCDW;

        public void Execute() {
            for (int n = 0; n < Outputs.Length; n++) {
                for (int w = 0; w < OutputsPrev.Length; w++) {
                    DCDW[n * OutputsPrev.Length + w] = DCDZ[n] * OutputsPrev[w];
                }
            }
        }
    }
}

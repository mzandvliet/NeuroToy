using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using System;
using System.Collections.Generic;

namespace NNBurst {
    public struct ConvLayer2D : System.IDisposable {
        public int KWidth;
        public int InWidth;
        public int OutWidth;

        public int InDepth;
        public int NumFilters;
        public int Stride;
        public int Padding;
        
        public NativeArray<float> Kernel;

        public NativeArray<float> Bias;
        public NativeArray<float> output;

        public static ConvLayer2D? Create(int inWidth, int inDepth, int kernWidth, int stride, int padding, int numFilters) {
            int outWidth = GetOutputWidth(inWidth, kernWidth, stride, padding);
            if (outWidth == -1) {
                Debug.LogError("Cannot perform convolution with this kernel, it is ill defined");
                return null;
            }
            return new ConvLayer2D(inWidth, inDepth, outWidth, kernWidth, stride, padding, numFilters);
        }        

        private ConvLayer2D(int inWidth, int inDepth, int outWidth, int kernWidth, int stride, int padding, int numFilters) {
            InWidth = inWidth;
            InDepth = inDepth;
            OutWidth = outWidth;
            KWidth = kernWidth;
            Stride = stride;
            Padding = padding;
            NumFilters = numFilters;
            Kernel = new NativeArray<float>(kernWidth * kernWidth * inDepth * numFilters, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            Bias = new NativeArray<float>(numFilters, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            output = new NativeArray<float>(outWidth * outWidth * numFilters, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }

        public void Dispose() {
            Kernel.Dispose();
            Bias.Dispose();
            output.Dispose();
        }

        // Todo: show user all possible integer solutions, given partially filled config
        public static int GetOutputWidth(int inWidth, int kernelWidth, int stride, int padding) {
            if (NeuralMath.IsEven(kernelWidth)) {
                return -1;
            }

            float result = (inWidth - kernelWidth + padding * 2.0f) / (float)stride + 1.0f;
            if (result - (int)result < float.Epsilon) {
                return (int)result;
            }
            return -1;
        }
    }

    public static class ConvolutionJobs {
        public static JobHandle ForwardPass(NativeArray<float> input, IList<ConvLayer2D> layers, JobHandle h) {
            for (int i = 0; i < layers.Count; i++) {
                h = ForwardPass(input, layers[i], h);
                input = layers[i].output;
            }

            return h;
        }
        public static JobHandle ForwardPass(NativeArray<float> input, ConvLayer2D layer, JobHandle h) {
            var cj = new Conv2DJob();
            cj.input = input;
            cj.layer = layer;
            h = cj.Schedule(h);

            var bj = new AddBias2DJob();
            bj.layer = layer;
            h = bj.Schedule(h);

            var rj = new NNBurst.ReluAssignJob();
            rj.Data = layer.output;
            h = rj.Schedule(h);

            return h;
        }

        // Todo: merge with the one from NeuralJobs (note: different activation function?)
        public static JobHandle ForwardPass(NativeArray<float> input, FCLayer layer, JobHandle h) {
            const int numThreads = 8;

            var b = new CopyParallelJob();
            b.From = layer.Biases;
            b.To = layer.Outputs;
            h = b.Schedule(layer.Outputs.Length, layer.Outputs.Length / numThreads, h);

            var d = new DotParallelJob();
            d.Input = input;
            d.Weights = layer.Weights;
            d.Output = layer.Outputs;
            h = d.Schedule(layer.Outputs.Length, layer.Outputs.Length / numThreads, h);

            var s = new SigmoidAssignParallelJob();
            s.Data = layer.Outputs;
            h = s.Schedule(layer.Outputs.Length, layer.Outputs.Length / numThreads, h);

            return h;
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
    }

    /*  
        Todo: split up into less monolithic functions
        Can do parallelism per filter slice, for example.
     */
    [BurstCompile]
    public struct Conv2DJob : IJob {
        [ReadOnly] public NativeArray<float> input;
        public ConvLayer2D layer;

        public void Execute() {
            var outSize = layer.OutWidth * layer.OutWidth;
            int kHalf = layer.KWidth / 2;
            int kSize = layer.KWidth * layer.KWidth;

            // For all filters
            for (int f = 0; f < layer.NumFilters; f++) {
                // Subset of memory corresponding to output for filter
                var output = layer.output.Slice(f * outSize, outSize);

                // Subset of memory corresponding to filter (with separate
                // values for each depth slice of the input
                int filterSpan = kSize * layer.InDepth; 
                var filter = layer.Kernel.Slice(f * filterSpan, filterSpan);

                // For all pixels in the output
                for (int x = 0; x < layer.OutWidth; x++) {
                    for (int y = 0; y < layer.OutWidth; y++) {
                        int inX = kHalf + x * layer.Stride;
                        int inY = kHalf + y * layer.Stride;

                        float act = 0f;

                        // For all depth slices in this filter
                        for (int fS = 0; fS < layer.InDepth; fS++) {
                            var fSlice = filter.Slice(fS * kSize, kSize);

                            // Dot product with kernel
                            for (int kX = -kHalf; kX <= kHalf; kX++) {
                                for (int kY = -kHalf; kY <= kHalf; kY++) {
                                    int inIdx = (inY + kY) * layer.InWidth + (inX + kX);
                                    int kernIdx = layer.KWidth * (kHalf + kY) + (kHalf + kX);

                                    act += input[inIdx] * fSlice[kernIdx];
                                }
                            }
                        }

                        output[y * layer.OutWidth + x] = act;
                    }
                }
            }
        }
    }

    [BurstCompile]
    public struct AddBias2DJob : IJob {
        public ConvLayer2D layer;

        public void Execute() {
            var outSize = layer.OutWidth * layer.OutWidth;

            for (int c = 0; c < layer.NumFilters; c++) {
                var o = layer.output.Slice(outSize * c, outSize);

                for (int i = 0; i < o.Length; i++) {
                    o[i] += layer.Bias[c];
                }
            }
        }
    }

    // Since ReLU calculations are orthogonal per pixel, no need for 2d structure
    [BurstCompile]
    public struct ReluAssignJob : IJob {
        public NativeArray<float> Data;

        public void Execute() {
            for (int i = 0; i < Data.Length; i++) {
                Data[i] = NeuralMath.ReLU(Data[i]);
            }
        }
    }

    [BurstCompile]
    public struct BackPropConvFinalJob : IJob {
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
    public struct BackPropConvJob : IJob {
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
    public struct BackPropConvWeightsJob : IJob {
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
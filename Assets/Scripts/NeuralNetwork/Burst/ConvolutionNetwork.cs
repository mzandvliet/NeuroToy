using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using System;

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
}
using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using System;

namespace NNBurst {
    public struct ConvLayer2D : System.IDisposable {
        public int Size;
        public int OutDepth;
        public int Stride;
        public int Padding;
        public int InDim;
        public int InDepth;
        public int OutDim;

        public NativeArray<float> Kernel;

        public NativeArray<float> Bias;
        public NativeArray<float> output;

        public static ConvLayer2D? Create(int inDim, int inDepth, int size, int outDepth, int stride, int padding) {
            int outDim = GetOutputSize(inDim, size, stride, padding);
            if (outDim == -1) {
                Debug.LogError("Cannot perform convolution with this kernel, output dimensions ill defined");
                return null;
            }
            return new ConvLayer2D(inDim, inDepth, outDim, size, outDepth, stride, padding);
        }        

        private ConvLayer2D(int inDim, int inDepth, int outDim, int size, int outDepth, int stride, int padding) {
            InDim = inDim;
            InDepth = inDepth;
            OutDim = outDim;
            Size = size;
            OutDepth = outDepth;
            Stride = stride;
            Padding = padding;
            Kernel = new NativeArray<float>(size * size * inDepth * outDepth, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            Bias = new NativeArray<float>(outDepth, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            output = new NativeArray<float>(outDim * outDim * outDepth, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }

        public void Dispose() {
            Kernel.Dispose();
            Bias.Dispose();
            output.Dispose();
        }

        private static int GetOutputSize(int inDim, int size, int stride, int padding) {
            float result = (inDim - size + padding * 2.0f) / (float)stride + 1.0f;
            if (result - (int)result < float.Epsilon) {
                return (int)result;
            }
            return -1;
        }
    }

    [BurstCompile]
    public struct Conv2DJob : IJob {
        [ReadOnly] public NativeArray<float> input;
        public ConvLayer2D layer;

        public void Execute() {
            var outDim = layer.OutDim;

            int kHalf = layer.Size / 2;

            for (int dOut = 0; dOut < layer.OutDepth; dOut++) {
                var o = layer.output.Slice(outDim * outDim * dOut, outDim * outDim);
                var k = layer.Kernel.Slice(layer.Size * layer.Size * dOut, layer.Size * layer.Size);

                /* 
                Todo: pass kernel over all input depth, since we now have
                separate weights for all n input depths.

                This means an additional inner loop over depthIn

                First, we need an easier way to address the data.
                It's all still dot products between vector subspaces.
                Find a few ways to make those way nicer to write.

                Slices are one way to make this nicer, since they at least
                encapsulate some addressing. This helps separate color
                channels, filter depth channels, whatever.

                Next, we want to have easier n-dimensional euclidean
                structure indexing.

                We could use functions to make this clearer, which compile
                to inlined code.

                Perhaps we can take the NativeSlice code and create our own
                EuclideanNativeSlice thing.
                */

                for (int x = 0; x < outDim; x += layer.Stride) {
                    for (int y = 0; y < outDim; y += layer.Stride) {
                        int inX = x + kHalf;
                        int inY = y + kHalf;

                        float a = 0f;

                        

                        for (int kX = -kHalf; kX <= kHalf; kX++) {
                            for (int kY = -kHalf; kY <= kHalf; kY++) {

                                int inIdx = (inY + kY) * layer.InDim + (inX + kX);
                                int kernIdx = layer.Size * (kHalf + kY) + (kHalf + kX);

                                a += input[inIdx] * k[kernIdx];
                            }
                        }

                        o[y * outDim + x] = a;
                    }
                }
            }
        }
    }

    [BurstCompile]
    public struct AdddBias2DJob : IJob {
        public ConvLayer2D layer;

        public void Execute() {
            var outDim = layer.OutDim;

            int kHalf = layer.Size / 2;

            for (int c = 0; c < layer.OutDepth; c++) {
                var o = layer.output.Slice(outDim * outDim * c, outDim * outDim);

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
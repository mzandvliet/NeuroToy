using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using System;

namespace NNBurst {
    public struct ConvLayer2D : System.IDisposable {
        public int Size;
        public int Depth;
        public int Stride;
        public int Padding;
        public int InDim;
        public int OutDim;

        public NativeArray<float> Kernel;

        public NativeArray<float> Bias;
        public NativeArray<float> output;

        public static ConvLayer2D? Create(int inDim, int size, int depth, int stride, int padding) {
            int outDim = GetOutputSize(inDim, size, stride, padding);
            if (outDim == -1) {
                Debug.LogError("Cannot perform convolution with this kernel, output dimensions ill defined");
                return null;
            }
            return new ConvLayer2D(inDim, outDim, size, depth, stride, padding);
        }        

        private ConvLayer2D(int inDim, int outDim, int size, int depth, int stride, int padding) {
            InDim = inDim;
            OutDim = outDim;
            Size = size;
            Depth = depth;
            Stride = stride;
            Padding = padding;
            Kernel = new NativeArray<float>(size * size * depth, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            Bias = new NativeArray<float>(depth, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            output = new NativeArray<float>(outDim * outDim * depth, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
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
        
        /* Todo:
         - Consider input has depth 16, this layer has depth 8; how to convolve?
         - Add bias (note: single bias per kernel, or bias per depth? Hmmmm...)
         - Padding
         - Multiple color channels
         - parallelize over depth or color channels (note: can't divvy up using NativeSlice)

         Note: bonus of having a single array for output: super easy to hook up FC layers
         */

        public void Execute() {
            var outDim = layer.OutDim;

            int kHalf = layer.Size / 2;

            for (int c = 0; c < layer.Depth; c++) {
                var o = layer.output.Slice(outDim * outDim * c, outDim * outDim);
                var k = layer.Kernel.Slice(layer.Size * layer.Size * c, layer.Size * layer.Size);

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

                        /* 
                        
                        Todo: separate job? Makes it more composable.
                        Separate job means:
                            - extra computation for looping :(
                            - easier composition and automatic differentiation :)

                        Best of both: compose as separate, but produce interleaved
                        job code for efficient computation. :D

                        Same goes for adding bias. Tensorflow has that separate.

                        It would make my backprop code look nicer.
                         
                         */

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

            for (int c = 0; c < layer.Depth; c++) {
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
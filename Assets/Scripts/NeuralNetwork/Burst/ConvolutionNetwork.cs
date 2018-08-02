using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using System;

namespace NNBurst {
    public struct ConvLayer2D : System.IDisposable {
        [ReadOnly] public int Size;
        [ReadOnly] public int Depth;
        [ReadOnly] public int Stride;
        [ReadOnly] public int Padding;
        [ReadOnly] public int InDim;
        [ReadOnly] public int OutDim;

        [ReadOnly] public NativeArray<float> Kernel;
        [WriteOnly] public NativeArray<float> output;

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
            output = new NativeArray<float>(outDim * outDim * depth, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }

        public void Dispose() {
            Kernel.Dispose();
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
                        int imgX = x + kHalf;
                        int imgY = y + kHalf;

                        float a = 0f;
                        for (int kX = -kHalf; kX <= kHalf; kX++) {
                            for (int kY = -kHalf; kY <= kHalf; kY++) {

                                int inIdx = (imgY + kY) * layer.InDim + (imgX + kX);
                                int kernIdx = layer.Size * (kHalf + kY) + (kHalf + kX);

                                a += input[inIdx] * k[kernIdx];
                            }
                        }

                        a = NeuralMath.ReLU(a);
                        o[y * outDim + x] = a;
                    }
                }
            }
        }
    }
}
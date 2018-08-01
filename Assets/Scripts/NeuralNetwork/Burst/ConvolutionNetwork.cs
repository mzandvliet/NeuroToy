using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using System;

namespace NNBurst {
    public struct Kernel2D : System.IDisposable {
        public int Size;
        public int Channels;
        public int Stride;

        public NativeArray<float> Values;

        public Kernel2D(int dims, int channels, int stride) {
            Size = dims;
            Channels = channels;
            Stride = stride;
            Values = new NativeArray<float>(dims * dims * channels, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }

        public void Dispose() {
            Values.Dispose();
        }
    }

    [BurstCompile]
    public struct Conv2DJob : IJob {
        [ReadOnly] public NativeArray<float> input;
        [WriteOnly] public NativeArray<float> output;
        [ReadOnly] public Kernel2D kernel;
        [ReadOnly] public int inDim;
        [ReadOnly] public int outDim;

        /* Todo:
         - Padding
         - Multiple color channels
         - parallelize over channels
         */

        public void Execute() {
            int kHalf = kernel.Size / 2;

            for (int c = 0; c < kernel.Channels; c++) {
                var o = output.Slice(outDim * outDim * c, outDim * outDim);
                var k = kernel.Values.Slice(kernel.Size * kernel.Size * c, kernel.Size * kernel.Size);

                for (int x = 0; x < outDim; x += kernel.Stride) {
                    for (int y = 0; y < outDim; y += kernel.Stride) {
                        int imgX = x + kHalf;
                        int imgY = y + kHalf;

                        float a = 0f;
                        for (int kX = -kHalf; kX <= kHalf; kX++) {
                            for (int kY = -kHalf; kY <= kHalf; kY++) {

                                int inIdx = (imgY + kY) * inDim + (imgX + kX);
                                int kIdx = kernel.Size * (kHalf + kY) + (kHalf + kX);

                                a += input[inIdx] * k[kIdx];
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
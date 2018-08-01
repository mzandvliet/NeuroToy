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
        [ReadOnly] public Kernel2D k;

        /* Todo:
         - experiment with native slice for notational clearity
         - parallelize over channels
         */

        public void Execute() {
            const int inDim = 28;
            const int outDim = inDim - 2; // Todo: derive from imgdim, kSize, kStride
            int kHalf = k.Size / 2;

            for (int c = 0; c < k.Channels; c++) {
                var outC = output.Slice(outDim * outDim * c, outDim * outDim);
                var kC = k.Values.Slice(k.Size * k.Size * c, k.Size * k.Size);

                for (int x = 0; x < outDim; x += k.Stride) {
                    for (int y = 0; y < outDim; y += k.Stride) {
                        int imgX = x + kHalf;
                        int imgY = y + kHalf;

                        float a = 0f;

                        for (int kX = -kHalf; kX <= kHalf; kX++) {
                            for (int kY = -kHalf; kY <= kHalf; kY++) {

                                int inIdx = (imgY + kY) * inDim + (imgX + kX);
                                int kIdx = k.Size * (kHalf + kY) + (kHalf + kX);

                                a += input[inIdx] * kC[kIdx];
                            }
                        }
                        outC[y * outDim + x] = a;
                    }
                }
            }
        }
    }
}
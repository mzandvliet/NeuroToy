using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;

namespace NNBurst {
    [BurstCompile]
    public struct Conv2DJob : IJob {
        [ReadOnly] public NativeArray<float> input;
        [ReadOnly] public NativeArray<float> kernel;
        [WriteOnly] public NativeArray<float> output;
        [ReadOnly] public int stride;

        // Todo: experiment with native slice architectures

        public void Execute() {
            for (int x = 0; x < 26; x+=stride) {
                for (int y = 0; y < 26; y+=stride) {
                    int imgX = x+1;
                    int imgY = y+1;

                    float act = 0f;
                    // act = input[imgY * 28 + imgX];
                    
                    for (int kX = -1; kX <= 1; kX++) {
                        for (int kY = -1; kY <= 1; kY++) {
                            act += input[(imgY+kY) * 28 + (imgX+kX)] * kernel[3 * (1 + kY) + (1 + kX)];
                        }
                    }
                    output[y * 26 + x] = act;
                }
            }
        }
    }
}
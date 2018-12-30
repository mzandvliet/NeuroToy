using Unity.Collections;
using Unity.Mathematics;
using Rng = Unity.Mathematics.Random;

namespace Ramjet.Mathematics {
    public static class Math {
        public const float Pi = 3.14159265359f;
        public const float Tau = Pi * 2f;
        public const float RadToDeg = 180f / Pi;

        public static float Sigmoid(float x) {
            return 1f / (1f + math.exp(-x));
        }

        public static float SigmoidD(float sigmoidX) {
            return sigmoidX * (1.0f - sigmoidX); // we already know sigmoid(x) from forward pass
        }

        public static float RandPolar(System.Random r) {
            return (float)(-1.0 + 2.0 * r.NextDouble());
        }

        // Todo: This is an expensive operation, we can optimize if needed
        // Todo: check range of random functions, inclusive vs. exclusive
        public static float Gaussian(ref Rng rng, float mean, float stddev) {
            float x1 = 1f - rng.NextFloat();
            float x2 = 1f - rng.NextFloat();

            float y1 = math.sqrt(-2.0f * math.log(x1)) * math.cos(2.0f * Pi * x2);
            return y1 * stddev + mean;
        }

        public static float2 RandPos(ref Rng rng, float2 boxMin, float2 boxMax) {
            return new float2(
                boxMin.x + (boxMax.x - boxMin.x) * rng.NextFloat(),
                boxMin.y + (boxMax.y - boxMin.y) * rng.NextFloat());
        }

        public static float WrapAngle(float a) {
            return a < 0 ? a + Tau : a > Tau ? a - Tau : a;
        }

        public static float AngleSigned(float2 a, float2 b) {
            float sin = a.x * b.y - b.x * a.y;
            float cos = a.x * b.x + a.y * b.y;
            return math.atan2(sin, cos) * RadToDeg;
        }

        public static float SignedPow(float v, float p) {
            return math.sign(v) * math.pow(v, p);
        }

        public static float2 SignedPow(float2 v, float p) {
            return new float2(math.sign(v.x) * math.pow(v.x, p), math.sign(v.y) * math.pow(v.y, p));
        }

        public static float2 GetUnitVector(float rotation) {
            return new float2(math.sin(rotation), math.cos(rotation));
        }

        /* Native Container Helpers */

        public static void FillConstant(NativeArray<float> array, float value) {
            for (int i = 0; i < array.Length; i++) {
                array[i] = value;
            }
        }
    }
}
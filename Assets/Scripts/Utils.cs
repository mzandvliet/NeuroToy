using System;
using UnityEngine;

public static class Utils {
    public const float Timestep = 0.016f;
    public const float Pi = 3.14159265359f;
    public const float Tau = Pi * 2f;
    public const float RadToDeg = 180f / Pi;


    public static void Copy(float[] source, float[] target) {
        if (source.Length != target.Length) {
            throw new ArgumentException("Arrays have to be same length");
        }

        for (int i = 0; i < source.Length; i++) {
            target[i] = source[i];
        }
    }

    public static void Zero(float[] nums) {
        Set(nums, 0f);
    }

    public static void Set(float[] nums, float value) {
        for (int i = 0; i < nums.Length; i++) {
            nums[i] = value;
        }
    }

    public static float Sigmoid(float x) {
        return 1f / (1f + Mathf.Exp(-x));
    }

    public static float SigmoidD(float sigmoidX) {
        return sigmoidX * Sigmoid(sigmoidX); // we already know sigmoid(x) from forward pass
    }

    public static float Tanh(float v) {
        return (float) Math.Tanh(v);
    }

    public static float RandPolar(System.Random r) {
        return (float)(-1.0 + 2.0 * r.NextDouble());
    }

    public static float Gaussian(System.Random r) {
        //uniform(0,1] random doubles
        double u1 = 1.0 - r.NextDouble();
        double u2 = 1.0 - r.NextDouble();
        //random normal(0,1)
        return (float)(
            Math.Sqrt(-2.0 * Math.Log(u1)) *
            Math.Sin(2.0 * Math.PI * u2));
    }

    // Todo: This is an expensive operation, we can optimize if needed
    public static float Gaussian(System.Random random, double mean, double stddev) {
        // The method requires sampling from a uniform random of (0,1]
        // but Random.NextDouble() returns a sample of [0,1).
        double x1 = 1 - random.NextDouble();
        double x2 = 1 - random.NextDouble();

        double y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
        return (float)(y1 * stddev + mean);
    }

    public static float Gaussian(float mean, float stddev) {
        // The method requires sampling from a uniform random of (0,1]
        // but Random.NextDouble() returns a sample of [0,1).
        float x1 = 1 - UnityEngine.Random.value;
        float x2 = 1 - UnityEngine.Random.value;

        float y1 = Mathf.Sqrt(-2.0f * Mathf.Log(x1)) * Mathf.Cos(2.0f * Mathf.PI * x2);
        return y1 * stddev + mean;
    }

    public static Vector2 RandPos(System.Random r, Vector2 boxMin, Vector2 boxMax) {
        return new Vector2(
            (float)(boxMin.x + (boxMax.x - boxMin.x) * r.NextDouble()),
            (float)(boxMin.y + (boxMax.y - boxMin.y) * r.NextDouble()));
    }

    public static float WrapAngle(float a) {
        return a < 0 ? a + Tau : a > Tau ? a - Tau : a;
    }

    public static float AngleSigned(Vector2 a, Vector2 b) {
        float sin = a.x * b.y - b.x * a.y;
        float cos = a.x * b.x + a.y * b.y;
        return Mathf.Atan2(sin, cos) * RadToDeg;
    }
  
    public static float SignedPow(float v, float p) {
        return Mathf.Sign(v) * Mathf.Pow(v, p);
    }

    public static Vector2 SignedPow(Vector2 v, float p) {
        return new Vector2(Mathf.Sign(v.x) * Mathf.Pow(v.x, p), Mathf.Sign(v.y) * Mathf.Pow(v.y, p));
    }

    public static Vector2 GetUnitVector(float rotation) {
        return new Vector2(Mathf.Sin(rotation), Mathf.Cos(rotation));
    }

    public static void GenerateTerrain(Terrain terrain) {
        float[,] heights = new float[terrain.terrainData.heightmapWidth, terrain.terrainData.heightmapHeight];
        for (int x = 0; x < terrain.terrainData.heightmapWidth; x++) {
            for (int y = 0; y < terrain.terrainData.heightmapWidth; y++) {
                float ruggedness = Mathf.Pow(y / (float) terrain.terrainData.heightmapHeight, 3f);
                heights[y, x] = 
                    ruggedness +
                    UnityEngine.Random.value * 0.003f * ruggedness;
            }
        }

        terrain.terrainData.SetHeights(0, 0, heights);
    }
}
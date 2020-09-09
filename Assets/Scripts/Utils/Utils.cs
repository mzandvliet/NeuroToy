using System;
using System.Collections.Generic;
using UnityEngine;
using Rng = Unity.Mathematics.Random;

namespace Ramjet {
    public static class Utils {
        public const float Timestep = 0.016f;

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

        public static void GenerateTerrain(Terrain terrain) {
            float[,] heights = new float[terrain.terrainData.heightmapResolution, terrain.terrainData.heightmapResolution];
            for (int x = 0; x < terrain.terrainData.heightmapResolution; x++) {
                for (int y = 0; y < terrain.terrainData.heightmapResolution; y++) {
                    float ruggedness = Mathf.Pow(y / (float)terrain.terrainData.heightmapResolution, 3f);
                    heights[y, x] =
                        ruggedness +
                        UnityEngine.Random.value * 0.003f * ruggedness;
                }
            }

            terrain.terrainData.SetHeights(0, 0, heights);
        }

        public static void Shuffle<T>(IList<T> list, ref Rng rng) {
            int n = list.Count;
            while (n > 1) {
                n--;
                int k = rng.NextInt(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }
    }
}
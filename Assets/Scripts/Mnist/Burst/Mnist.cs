using System.IO;
using UnityEngine;
using Unity.Collections;
using System.Collections.Generic;
using Unity.Jobs;
using Rng = Unity.Mathematics.Random;
using Ramjet.Math.FixedPoint;
using Unity.Collections.LowLevel.Unsafe;

namespace NNBurst.Mnist {
    public struct DatasetFloat : System.IDisposable {
        public NativeArray<int> Labels;
        public NativeArray<float> Images;
        public List<int> Indices;

        public int NumImgs {
            get { return Labels.Length; }
        }

        public DatasetFloat(int count) {
            Labels = new NativeArray<int>(count, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            Images = new NativeArray<float>(count * DataManager.ImgDims * DataManager.Channels, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            Indices = new List<int>(count);
        }

        public void Dispose() {
            Labels.Dispose();
            Images.Dispose();
        }
    }

    public struct DatasetByte : System.IDisposable {
        public NativeArray<int> Labels;
        public NativeArray<byte> Images;
        public List<int> Indices;

        public int NumImgs {
            get { return Labels.Length; }
        }

        public DatasetByte(int count) {
            Labels = new NativeArray<int>(count, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            Images = new NativeArray<byte>(count * DataManager.ImgDims * DataManager.Channels, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            Indices = new List<int>(count);
        }

        public void Dispose() {
            Labels.Dispose();
            Images.Dispose();
        }
    }

    public static class DataManager {
        private const string Folder = "./Datasets/Mnist";
        private const string TrainImagePath = "train-images.idx3-ubyte";
        private const string TrainLabelPath = "train-labels.idx1-ubyte";
        private const string TestImagePath = "t10k-images.idx3-ubyte";
        private const string TestLabelPath = "t10k-labels.idx1-ubyte";

        public static DatasetFloat TrainFloats;
        public static DatasetFloat TestFloats;

        public static DatasetByte TrainBytes;
        public static DatasetByte TestBytes;

        public const int Width = 28;
        public const int Height = 28;
        public const int ImgDims = Width * Height;
        public const int Channels = 1;

        public static void Load() {
            TrainFloats = LoadAsFloats(TrainImagePath, TrainLabelPath);
            TestFloats = LoadAsFloats(TestImagePath, TestLabelPath);

            TrainBytes = LoadAsBytes(TrainImagePath, TrainLabelPath);
            TestBytes = LoadAsBytes(TestImagePath, TestLabelPath);
        }

        public static void Unload() {
            Debug.Log("MNIST: Unloading datasets...");
            TrainFloats.Dispose();
            TestFloats.Dispose();
        }

        private static DatasetFloat LoadAsFloats(string imgPath, string lblPath) {
            var lblReader = new BigEndianBinaryReader(new FileStream(Path.Combine(Folder, lblPath), FileMode.Open));
            var imgReader = new BigEndianBinaryReader(new FileStream(Path.Combine(Folder, imgPath), FileMode.Open));

            lblReader.ReadInt32();
            lblReader.ReadInt32();

            int magicNum = imgReader.ReadInt32();
            int NumImgs = imgReader.ReadInt32();
            int Rows = imgReader.ReadInt32();
            int Cols = imgReader.ReadInt32();
            int ImgDims = Rows * Cols;

            Debug.Log("MNIST: Loading " + imgPath + ", Imgs: " + NumImgs + " Rows: " + Rows + " Cols: " + Cols);

            var set = new DatasetFloat(NumImgs);

            for (int i = 0; i < NumImgs; i++) {
                byte lbl = lblReader.ReadByte();
                set.Labels[i] = (int)lbl;
            }

            for (int i = 0; i < NumImgs; i++) {
                // Read order flips images axes to Unity style
                for (int y = 0; y < Cols; y++) {
                    for (int x = 0; x < Rows; x++) {
                        byte pix = imgReader.ReadByte();
                        set.Images[i * ImgDims + (Cols - 1 - y) * Rows + x] = pix / 256f;
                    }
                }
            }

            lblReader.Close();
            imgReader.Close();

            return set;
        }

        private static DatasetByte LoadAsBytes(string imgPath, string lblPath) {
            var lblReader = new BigEndianBinaryReader(new FileStream(Path.Combine(Folder, lblPath), FileMode.Open));
            var imgReader = new BigEndianBinaryReader(new FileStream(Path.Combine(Folder, imgPath), FileMode.Open));

            lblReader.ReadInt32();
            lblReader.ReadInt32();

            int magicNum = imgReader.ReadInt32();
            int NumImgs = imgReader.ReadInt32();
            int Rows = imgReader.ReadInt32();
            int Cols = imgReader.ReadInt32();
            int ImgDims = Rows * Cols;

            Debug.Log("MNIST: Loading " + imgPath + ", Imgs: " + NumImgs + " Rows: " + Rows + " Cols: " + Cols);

            var set = new DatasetByte(NumImgs);

            for (int i = 0; i < NumImgs; i++) {
                byte lbl = lblReader.ReadByte();
                set.Labels[i] = (int)lbl;
            }

            for (int i = 0; i < NumImgs; i++) {
                // Read order flips images axes to Unity style
                for (int y = 0; y < Cols; y++) {
                    for (int x = 0; x < Rows; x++) {
                        byte pix = imgReader.ReadByte();
                        set.Images[i * ImgDims + (Cols - 1 - y) * Rows + x] = pix;
                    }
                }
            }

            lblReader.Close();
            imgReader.Close();

            return set;
        }

        public static void GetBatch(NativeArray<int> batch, DatasetFloat set, ref Rng rng) {
            // Todo: can transform dataset to create additional variation

            UnityEngine.Profiling.Profiler.BeginSample("GetBatch");

            if (set.Indices.Count < batch.Length) {
                set.Indices.Clear();
                for (int i = 0; i < set.NumImgs; i++) {
                    set.Indices.Add(i);
                }
                Ramjet.Utils.Shuffle(set.Indices, ref rng);
            }

            for (int i = 0; i < batch.Length; i++) {
                batch[i] = set.Indices[set.Indices.Count - 1];
                set.Indices.RemoveAt(set.Indices.Count - 1);
            }

            UnityEngine.Profiling.Profiler.EndSample();
        }

        public static void GetBatch(NativeArray<int> batch, DatasetByte set, ref Rng rng) {
            // Todo: can transform dataset to create additional variation

            UnityEngine.Profiling.Profiler.BeginSample("GetBatch");

            if (set.Indices.Count < batch.Length) {
                set.Indices.Clear();
                for (int i = 0; i < set.NumImgs; i++) {
                    set.Indices.Add(i);
                }
                Ramjet.Utils.Shuffle(set.Indices, ref rng);
            }

            for (int i = 0; i < batch.Length; i++) {
                batch[i] = set.Indices[set.Indices.Count - 1];
                set.Indices.RemoveAt(set.Indices.Count - 1);
            }

            UnityEngine.Profiling.Profiler.EndSample();
        }

        public static JobHandle CopyInput(NativeArray<float> inputs, NNBurst.Mnist.DatasetFloat set, int imgIdx, JobHandle handle = new JobHandle()) {
            var copyInputJob = new CopySubsetJob<float>();
            copyInputJob.From = set.Images;
            copyInputJob.To = inputs;
            copyInputJob.Length = ImgDims;
            copyInputJob.FromStart = imgIdx * ImgDims;
            copyInputJob.ToStart = 0;
            return copyInputJob.Schedule(handle);
        }

        public static void CopyBytesToInput(NativeArray<byte> inputs, NNBurst.Mnist.DatasetByte set, int imgIdx) {
            set.Images.Slice(imgIdx * ImgDims, ImgDims).CopyTo(inputs);
        }

        public static void ToTexture(DatasetFloat set, int imgIndex, Texture2D tex) {
            var colors = new Color[ImgDims];

            for (int y = 0; y < Height; y++) {
                for (int x = 0; x < Width; x++) {
                    float pix = set.Images[imgIndex * ImgDims + y * Height + x];
                    colors[y * Height + x] = new Color(pix, pix, pix, 1f);
                }
            }

            tex.SetPixels(0, 0, Width, Height, colors);
            tex.Apply(false);
        }
    }
}